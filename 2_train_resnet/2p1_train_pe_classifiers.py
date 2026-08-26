#!/usr/bin/env python3
"""Train PE [pulmonary embolism] detection classifiers on real / fake / balanced RSNA-PE data.
"""
from __future__ import annotations
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import zipfile
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import AUROC, Accuracy
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)   # so `backbone` resolves whether this is run as a
                                     # script (python 2p1_....py) or as -m 2_train_resnet.2p1_...
from backbone import build_classifier

CONFIG_PATH = os.path.join(SCRIPT_DIR, "pe_train_config_resnet50.yml")

UID_COLS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]

# RSNA-PE DICOM slices are always 512x512 before resizing to the training resolution `px`.
DICOM_TARGET_SIZE = (512, 512)
# build_augment's RandomResizedCrop scale/aspect-ratio jitter range (mild, close to 1:1).
AUG_SCALE_RANGE = (0.85, 1.0)
AUG_RATIO_RANGE = (0.95, 1.05)
# build_augment's occasional (p=AUG_PROB) resize-rotate-crop branch.
RESIZE_DIM = 290
ROTATION_DEGREES = 15
AUG_PROB = 0.7



def get_dicom_path(uids: Sequence[str]) -> str:
    return "train/" + "/".join(uids) + ".dcm"


def get_dicom_array(archive: zipfile.ZipFile, path: str, window: tuple[float, float]) -> np.ndarray:
    dcm = dicom.dcmread(archive.open(path))
    arr = dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept
    return arr.clip(*window)


def load_rsna_rows(
        df: pd.DataFrame, archive: zipfile.ZipFile, row_ids: np.ndarray,
        window: tuple[float, float], px: int
) -> torch.Tensor:
    """Load + resize the DICOM slices at metadata rows `row_ids` -> (n, 1, px, px) float32."""
    imgs = torch.zeros(len(row_ids), *DICOM_TARGET_SIZE)
    for k, idx in enumerate(tqdm(row_ids, desc="loading RSNA-PE slices", unit="img")):
        r = df.loc[idx]
        path = get_dicom_path([r.StudyInstanceUID, r.SeriesInstanceUID, r.SOPInstanceUID])
        arr = get_dicom_array(archive, path, window)
        assert arr.shape == DICOM_TARGET_SIZE
        imgs[k] = torch.from_numpy(arr.astype("float32"))
    return F.interpolate(imgs.unsqueeze(1), size=px, mode="nearest")


def collect_row_range_slices(
        df: pd.DataFrame, archive: zipfile.ZipFile, row_lo: int, row_hi: int,
        window: tuple[float, float], px: int, max_neg: int, max_pos: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evenly-capped neg/pos slices drawn from metadata rows [row_lo, row_hi).
    """
    sub = df.iloc[row_lo:row_hi]
    pos_rows = sub.index[sub["pe_present_on_image"] == 1].to_numpy()
    neg_rows = sub.index[sub["pe_present_on_image"] == 0].to_numpy()

    rng = np.random.default_rng(seed)
    pos_rows = rng.choice(pos_rows, size=min(max_pos, len(pos_rows)), replace=False)
    neg_rows = rng.choice(neg_rows, size=min(max_neg, len(neg_rows)), replace=False)

    neg = load_rsna_rows(df, archive, neg_rows, window, px)
    pos = load_rsna_rows(df, archive, pos_rows, window, px)
    return neg, pos


def build_augment(px: int) -> torch.nn.Sequential:
    """Geometric augmentation applied per training image (not validation)."""
    return torch.nn.Sequential(
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomResizedCrop(px, scale=AUG_SCALE_RANGE, ratio=AUG_RATIO_RANGE, antialias=True),
        T.RandomApply(torch.nn.ModuleList([
            T.Resize(RESIZE_DIM, antialias=True),
            T.RandomRotation(ROTATION_DEGREES),
            T.CenterCrop(px),
        ]), p=AUG_PROB),
    )


class SliceDataset(Dataset):
    """(image, label) pairs read lazily from a memmap, then augment + clip + normalise.
    """

    def __init__(
            self, images: np.ndarray, labels: torch.Tensor, clip: tuple[float, float],
            mean: float, std: float, augment: Optional[Callable] = None
    ) -> None:
        self.images, self.labels = images, labels.long()
        self.clip, self.mean, self.std = clip, mean, std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.images[i]))   # copy one slice off the memmap
        if self.augment is not None:
            x = self.augment(x)
        x = x.clamp(self.clip[0], self.clip[1])
        return (x - self.mean) / self.std, self.labels[i]


def make_balanced_mixed(
        real_ds: Dataset, real_labels: torch.Tensor, fake_ds: Dataset, fake_labels: torch.Tensor,
        per_class: int, generator: torch.Generator
) -> Subset:
    """
    Balanced dataset: `per_class` samples per class from its real+fake pool.
    """
    concat = ConcatDataset([real_ds, fake_ds])
    offset = len(real_ds)                       # fake indices are shifted by len(real)

    def pool(label: int) -> torch.Tensor:
        real_i = (real_labels == label).nonzero().squeeze(1)
        fake_i = (fake_labels == label).nonzero().squeeze(1) + offset
        idx = torch.cat([real_i, fake_i])
        return idx[torch.randperm(len(idx), generator=generator)[:per_class]]

    chosen = torch.cat([pool(0), pool(1)])
    return Subset(concat, chosen[torch.randperm(len(chosen), generator=generator)].tolist())


def cached_slices(
        cache_dir: str, name: str, builder: Callable[[], tuple[torch.Tensor, torch.Tensor]]
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Return (images_memmap, labels_tensor) for `name`, building + caching on first use.
    """
    img_path = os.path.join(cache_dir, f"{name}_images.npy")
    lbl_path = os.path.join(cache_dir, f"{name}_labels.npy")
    if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
        os.makedirs(cache_dir, exist_ok=True)
        images, labels = builder()                       # tensors (materialised once)
        np.save(img_path, images.numpy())
        np.save(lbl_path, labels.numpy())
        del images, labels                               # free before memmapping back
    return np.load(img_path, mmap_mode="r"), torch.from_numpy(np.load(lbl_path))


def build_pools(
        cfg: dict
) -> tuple[Dataset, torch.Tensor, Optional[Dataset], Optional[torch.Tensor], Dataset]:
    """Memmap the real-train / fake / val splits (caching real & val on first run)."""
    p, d = cfg["paths"], cfg["data"]
    clip, mean, std, px, window = d["clip"], d["norm_mean"], d["norm_std"], d["px"], d["window"]
    cache_dir = p.get("cache_dir", os.path.join(p["output_dir"], "cache"))
    augment = build_augment(px) if d.get("augment", True) else None

    df = pd.read_csv(p["metadata_csv"], usecols=UID_COLS + ["pe_present_on_image"])
    archive = zipfile.ZipFile(p["zip_path"], "r")

    def build_split(rows: tuple[int, int], caps: dict, seed_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect a row-range split -> (images, labels) tensors, shuffled once with a fixed
        seed so neg/pos aren't stored as class-contiguous blocks."""
        neg, pos = collect_row_range_slices(df, archive, rows[0], rows[1], window, px,
                                            caps["neg"], caps["pos"], cfg["seed"] + seed_offset)
        imgs = torch.cat([neg, pos])
        lbls = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))])
        order = torch.randperm(len(imgs), generator=torch.Generator().manual_seed(cfg["seed"] + seed_offset))
        return imgs[order], lbls[order]

    r, v = d["real_train"], d["real_val"]
    real_imgs, real_lbls = cached_slices(cache_dir, f"real_{r['neg']}n_{r['pos']}p_{px}px",
                                         lambda: build_split(d["real_train_rows"], r, 0))
    val_imgs, val_lbls = cached_slices(cache_dir, f"val_{v['neg']}n_{v['pos']}p_{px}px",
                                       lambda: build_split(d["real_val_rows"], v, 1))

    real_ds = SliceDataset(real_imgs, real_lbls, clip, mean, std, augment)
    val_ds = SliceDataset(val_imgs, val_lbls, clip, mean, std, augment=None)

    # Fake slices are only needed for the 'fake' and 'balanced' variants -- skip loading them
    # (and don't require the file to exist) otherwise.
    fake_ds = fake_lbls = None
    if any(var in cfg["variants"] for var in ("fake", "balanced")):
        fake_imgs = np.load(p["fake_images"], mmap_mode="r")   # one preprocessed .npy -> memmap
        fake_lbls = torch.from_numpy(np.load(p["fake_labels"]))
        fake_ds = SliceDataset(fake_imgs, fake_lbls, clip, mean, std, augment)

    fake_info = f"fake: {len(fake_ds)} (pos {int(fake_lbls.sum())})" if fake_ds is not None else "fake: (not loaded)"
    print(f"real: {len(real_ds)} (pos {int(real_lbls.sum())}) | {fake_info} | val: {len(val_ds)}")
    return real_ds, real_lbls, fake_ds, fake_lbls, val_ds


# =========================================================================== #
# Model: a LightningModule wrapping the configurable backbone
# =========================================================================== #
class PEClassifier(pl.LightningModule):
    """Binary PE classifier: weighted cross-entropy, per-epoch LR decay, acc / AUROC."""

    def __init__(
            self, model_cfg: dict, base_lr: float, weight_decay: float,
            class_weights: torch.Tensor, pct_start: float = 0.1
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = build_classifier(**model_cfg)
        self.register_buffer("class_weights", class_weights)
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_auroc = AUROC(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.train_acc(logits, y)
        self.log_dict({"train/loss": loss, "train/acc": self.train_acc},
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.val_acc(logits, y)
        self.val_auroc(logits.softmax(1)[:, 1], y)
        self.log_dict({"val/loss": loss, "val/acc": self.val_acc, "val/auroc": self.val_auroc},
                      on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.base_lr, weight_decay=self.hparams.weight_decay)
        # One-cycle: linear warmup for `pct_start` of training, then anneal down 
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.base_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.pct_start, div_factor=10, final_div_factor=100,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# =========================================================================== #
# Training driver
# =========================================================================== #
def make_loggers(cfg: dict, run_name: str, variant: str) -> list:
    out, wb = cfg["paths"]["output_dir"], cfg["logging"]["wandb"]
    loggers = [WandbLogger(project=wb["project"], entity=wb["entity"], name=run_name,
                           save_dir=out, mode=wb.get("mode", "online"),
                           group=variant)]   # so the N runs of each variant can be overlaid in the UI
    if cfg["logging"].get("csv", True):
        loggers.append(CSVLogger(save_dir=out, name=run_name))
    return loggers


def train_once(variant: str, run: int, train_ds: Dataset, val_ds: Dataset, cfg: dict) -> float:
    """Train one (variant, run) and return the best validation AUROC."""
    t, run_name = cfg["training"], f"{variant}_run{run}"
    print(f"\n===== {run_name} =====")
    weights = torch.tensor(t["class_weights"][variant], dtype=torch.float32)
    model = PEClassifier(cfg["model"], t["base_lr"], t["weight_decay"], weights,
                         t.get("pct_start", 0.1))

    loader_kw = dict(batch_size=t["batch_size"], num_workers=cfg["data"]["num_workers"],
                     pin_memory=True,
                     # Without this, the worker pool is torn down and respawned on every single
                     # iteration over the loader 
                     persistent_workers=cfg["data"]["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # Keep the best-val-AUROC checkpoint 
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(cfg["paths"]["output_dir"], run_name),
                              monitor="val/auroc", mode="max", save_top_k=1,
                              filename="best-{epoch}-{step}")
    stop_cb = EarlyStopping(monitor="val/auroc", mode="max",
                            patience=t.get("early_stopping_patience", 30))
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=t["max_epochs"],
        limit_train_batches=t["limit_train_batches"],
        val_check_interval=t["val_check_interval"],
        limit_val_batches=t.get("limit_val_batches", 1.0),
        accelerator=t["accelerator"], devices=t["devices"], precision=t["precision"],
        logger=make_loggers(cfg, run_name, variant), callbacks=[ckpt_cb, stop_cb, lr_cb],
        default_root_dir=cfg["paths"]["output_dir"], log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    best = float(ckpt_cb.best_model_score)
    print(f"{run_name}: best val/auroc = {best:.4f}  ({ckpt_cb.best_model_path})")
    wandb.finish()
    return best


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG_PATH))

    pl.seed_everything(cfg["seed"], workers=True)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    real_ds, real_lbls, fake_ds, fake_lbls, val_ds = build_pools(cfg)

    # Repeated runs; the balanced set is re-drawn per run (different seed).
    results = {v: [] for v in cfg["variants"]}
    for run in range(cfg["n_runs"]):
        variant_ds = {"real": real_ds, "fake": fake_ds}
        if "balanced" in cfg["variants"]:   # only build the balanced mix if it's requested
            gen = torch.Generator().manual_seed(cfg["seed"] + run)
            variant_ds["balanced"] = make_balanced_mixed(real_ds, real_lbls, fake_ds, fake_lbls,
                                                         cfg["data"]["balanced_per_class"], gen)
        for variant in cfg["variants"]:
            results[variant].append(train_once(variant, run, variant_ds[variant], val_ds, cfg))

    print("\n===== mean best val/auroc over runs =====")
    for variant, scores in results.items():
        print(f"  {variant:8s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
