#!/usr/bin/env python3
"""
Extract and cache RSNA-PE [pulmonary embolism] latent codes.
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------- #
# Project paths / config
# --------------------------------------------------------------------------- #
# Anchor to the repo root (not the caller's cwd), since this script lives one level down, in
# 1_generateData/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "1p1_extract_latent_codes_config.yml")

import yaml  # noqa: E402  (import after computing CONFIG_PATH, before the env-var-sensitive imports below)

with open(CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

# Set before importing torch, so CUDA_VISIBLE_DEVICES actually takes effect.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if _cfg.get("device_id") is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["device_id"])

import sys
import threading
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Autoencoder  # noqa: E402  (import after sys.path setup)

METADATA_CSV = _cfg["metadata_csv"]
ZIP_PATH     = _cfg["zip_path"]
CKPT         = os.path.join(REPO_ROOT, _cfg["ckpt"])
OUT_DIR      = os.path.join(REPO_ROOT, _cfg["out_dir"])
ROW_LO, ROW_HI = _cfg["row_lo"], _cfg["row_hi"]

UID_COLS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
WINDOW = (-1000, 3000)   # HU [Hounsfield unit] clip range -- covers air through dense bone
PX     = 256             # slice resolution after downsampling
LAT    = 512             # latent code dimensionality


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def get_dicom_path(uids: Sequence[str]) -> str:
    return "train/" + "/".join(uids) + ".dcm"


_thread_local = threading.local()


def get_archive(zip_path: str) -> zipfile.ZipFile:
    """One ZipFile handle per thread -- zipfile.ZipFile isn't safe to share across threads."""
    archive = getattr(_thread_local, "archive", None)
    if archive is None:
        archive = zipfile.ZipFile(zip_path, "r")
        _thread_local.archive = archive
    return archive


def load_slice(row, zip_path: str, window, px: int) -> torch.Tensor:
    """Load + window + downsample one metadata row's DICOM slice -> (1, px, px) float32."""
    archive = get_archive(zip_path)
    path = get_dicom_path([row.StudyInstanceUID, row.SeriesInstanceUID, row.SOPInstanceUID])
    dcm = dicom.dcmread(archive.open(path))
    arr = (dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept).clip(*window)
    assert arr.shape == (512, 512)
    t = torch.from_numpy(arr.astype("float32")).unsqueeze(0).unsqueeze(0)   # (1, 1, 512, 512)
    return F.interpolate(t, size=px, mode="nearest").squeeze(0)             # (1, px, px)


def prefetch(items: Sequence, loader: Callable, workers: int, ahead: int) -> Iterator:
    """Yield ``(item, loader(item))`` in order while loading `ahead` items concurrently.

    Overlaps disk I/O + DICOM decode with GPU compute: while the consumer encodes one
    batch, background threads are already loading the next several slices.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        it = iter(items)
        for _ in range(ahead):                             # prime the pipeline
            try:
                x = next(it)
            except StopIteration:
                break
            pending.append((x, pool.submit(loader, x)))
        while pending:
            x, future = pending.popleft()
            try:                                           # keep the queue full
                nxt = next(it)
                pending.append((nxt, pool.submit(loader, nxt)))
            except StopIteration:
                pass
            yield x, future.result()


def chunked(iterable: Iterator, n: int) -> Iterator[list]:
    """Group an iterable into lists of (up to) `n` items."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# --------------------------------------------------------------------------- #
# Model / encoding
# --------------------------------------------------------------------------- #
def load_model(ckpt: str, device: str) -> Autoencoder:
    model = Autoencoder(PX, (64, 128, 256, 512, 1024), (32, 64, 128, 256, 512),
                        3, LAT, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval()


def compute_norm_stats(rows, zip_path: str, window, px: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std estimated from a handful of representative slices."""
    sample = torch.stack([load_slice(r, zip_path, window, px) for r in rows.itertuples(index=False)],
                         dim=0)   # (n, 1, px, px)
    return sample.mean(dim=(0, 2, 3)), sample.std(dim=(0, 2, 3))


@torch.no_grad()
def extract_codes(rows, zip_path: str, model: Autoencoder, mean: torch.Tensor, std: torch.Tensor,
                  device: str, window, px: int, batch_size: int, workers: int, prefetch_n: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream all `rows`, encoding every slice into a code with its PE label."""
    normalize = T.Normalize(mean, std)
    loader = lambda row: load_slice(row, zip_path, window, px)
    items = list(rows.itertuples(index=False))

    code_chunks, label_chunks = [], []
    pbar = tqdm(total=len(items), desc="encoding", unit="img")
    for batch in chunked(prefetch(items, loader, workers, prefetch_n), batch_size):
        batch_rows, batch_slices = zip(*batch)
        x = normalize(torch.stack(batch_slices, dim=0)).to(device, non_blocking=True)
        code_chunks.append(model.classif(x).cpu())
        label_chunks.append(torch.tensor([float(r.pe_present_on_image) for r in batch_rows]))
        pbar.update(len(batch))
    pbar.close()

    return torch.cat(code_chunks, dim=0), torch.cat(label_chunks, dim=0)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.backends.cudnn.benchmark = True   # fixed-ish input sizes -> faster convs

    os.makedirs(OUT_DIR, exist_ok=True)
    codes_path  = os.path.join(OUT_DIR, "train_codes.npy")
    labels_path = os.path.join(OUT_DIR, "train_code_labels.npy")
    stats_path  = os.path.join(OUT_DIR, "norm_mean_std.npy")
    if os.path.exists(codes_path) and not _cfg["force"]:
        sys.exit(f"{codes_path} already exists -- set force: true in "
                 f"{os.path.basename(CONFIG_PATH)} to overwrite.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA unavailable; falling back to CPU.")

    print("Loading model and metadata ...")
    model = load_model(CKPT, device)
    df = pd.read_csv(METADATA_CSV, usecols=UID_COLS + ["pe_present_on_image"])
    rows = df.iloc[ROW_LO:ROW_HI]
    if _cfg["limit"]:
        rows = rows.iloc[:_cfg["limit"]]
    n_pos = int(rows["pe_present_on_image"].sum())
    print(f"Row range [{ROW_LO}, {ROW_HI}): {len(rows):,} slices "
          f"({n_pos:,} PE+, {len(rows) - n_pos:,} PE-)")

    # Normalisation stats from a small, evenly-spread sample of slices in the row range.
    sample_pos = np.linspace(0, len(rows) - 1, min(_cfg["stats_rows"], len(rows))).astype(int)
    mean, std = compute_norm_stats(rows.iloc[sample_pos], ZIP_PATH, WINDOW, PX)
    print(f"Normalisation: mean={mean.item():.2f}, std={std.item():.2f} "
          f"(from {len(sample_pos)} sampled slices)")

    codes, labels = extract_codes(rows, ZIP_PATH, model, mean, std, device, WINDOW, PX,
                                  _cfg["batch_size"], _cfg["workers"], _cfg["prefetch"])

    np.save(codes_path,  codes.numpy())
    np.save(labels_path, labels.numpy())
    np.save(stats_path,  np.stack([mean.numpy(), std.numpy()]))
    print(f"\nSaved {codes.shape[0]:,} codes "
          f"({int((labels == 1).sum()):,} PE+, {int((labels == 0).sum()):,} PE-)")
    print(f"  {codes_path}   {tuple(codes.shape)}")
    print(f"  {labels_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
