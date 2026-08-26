#!/usr/bin/env python3
"""Generate a synthetic (fake) RSNA-PE CT dataset from the latent codes
extracted by 1p1_extract_latent_codes.py.

Pipeline:
    1. Load the saved per-slice latent codes + labels + normalisation stats.
    2. Fit a per-class Gaussian KDE [kernel density estimate] over the codes
       (PE-negative / PE-positive).
    3. Resample codes from each KDE and REJECT any that land too close (squared
       L2 < threshold) to a real training code
    4. Decode the accepted codes into images with the diffusion autoencoder from stage 0
       (0_trainDiffAE/), using classifier-free guidance [CFG] sampling.
    5. Save the images and their binary PE labels.

Inputs  (from CODES_DIR, produced by 1p1_extract_latent_codes.py):
    train_codes.npy, train_code_labels.npy, norm_mean_std.npy
Outputs (to OUT_DIR):
    fake_images.npy   float32 (N, 1, 256, 256)
    fake_labels.npy   float32 (N,)   -- 0.0 = PE-negative, 1.0 = PE-positive

"""
from __future__ import annotations

import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "1p2_generate_fake_dataset_config.yml")

with open(CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

# Set before importing torch, so CUDA_VISIBLE_DEVICES actually takes effect.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if _cfg.get("device_id") is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["device_id"])

import numpy as np
import scipy.stats
import torch
from tqdm import tqdm


CKPT      = os.path.join(REPO_ROOT, _cfg["ckpt"])         # autoencoder checkpoint from stage 0
DEVICE    = _cfg["device"]                                  # torch device ("cuda:0" or "cpu")
CODES_DIR = os.path.join(REPO_ROOT, _cfg["codes_dir"])    # dir holding the cached codes
OUT_DIR   = _cfg["out_dir"]                                 # where fake_images/labels.npy go

N_NEG = _cfg["n_neg"]   # number of PE-negative fakes to generate
N_POS = _cfg["n_pos"]   # number of PE-positive fakes to generate

KDE_BANDWIDTH = _cfg["kde_bandwidth"]   # gaussian_kde bw_method for both classes

REJECT_THRESHOLD  = _cfg["reject_threshold"]
REJECT_PERCENTILE = _cfg["reject_percentile"]   # percentile of real NN-distance used when auto-deriving
NN_SAMPLE_SIZE    = _cfg["nn_sample_size"]      # #real codes sampled to estimate the NN-distance percentiles

N_CANDIDATES = _cfg["n_candidates"]   # codes resampled from the KDE per batch
BATCH_CAP    = _cfg["batch_cap"]        # max accepted codes decoded per diffusion batch

CFG_SCALE       = _cfg["cfg_scale"]         # classifier-free guidance scale
ANGLE_STEP      = _cfg["angle_step"]        # diffusion angle step
NOISE_RETENTION = _cfg["noise_retention"]   # diffusion noise retention

SHUFFLE = _cfg["shuffle"]   # shuffle the combined dataset before saving
SEED    = _cfg["seed"]      # RNG seed for reproducible sampling (None = nondeterministic)
FORCE   = _cfg["force"]     # overwrite existing outputs
# ==========================================================================

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.diffusion import Diffusion   # noqa: E402  (import after sys.path setup)
from src.models import Autoencoder    # noqa: E402

PX  = 256          # image resolution
LAT = 512          # latent code dimensionality

def load_model(ckpt: str, device: str) -> Autoencoder:
    model = Autoencoder(PX, (64, 128, 256, 512, 1024), (32, 64, 128, 256, 512),
                        3, LAT, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval()


def load_diffusion(stats_path: str, device: str) -> Diffusion:
    """Build the diffusion sampler with the saved normalisation mean/std."""
    diffusion = Diffusion(img_size=PX, device=device, lat_size=LAT)
    mean, std = np.load(stats_path)            # shape (2, 1)
    diffusion.mean = float(mean[0])
    diffusion.std = float(std[0])
    return diffusion


def fit_kdes(codes: torch.Tensor, labels: torch.Tensor, bw: float):
    """Fit a Gaussian KDE over the codes of each class. Returns (neg_kde, pos_kde)."""
    neg = codes[labels == 0].numpy()
    pos = codes[labels == 1].numpy()
    print(f"Fitting KDEs on {len(neg):,} PE- and {len(pos):,} PE+ codes (bw={bw}) ...")
    return (scipy.stats.gaussian_kde(neg.T, bw_method=bw),
            scipy.stats.gaussian_kde(pos.T, bw_method=bw))


def estimate_nn_threshold(codes: torch.Tensor, device: str, percentile: float,
                          sample_size: int, chunk: int = 50000) -> float:
    """Estimate a rejection threshold from REAL-to-REAL squared-L2 nearest-neighbour
    distances, so the memorisation guard is calibrated to this dataset's own code
    geometry instead of a constant borrowed from a different dataset (see module
    docstring). Returns the given percentile of those distances.
    """
    ref = codes.to(device)
    n = len(ref)
    idx = torch.randperm(n, device=device)[:min(sample_size, n)]
    q = ref[idx]

    nn_d2 = torch.full((len(idx),), float("inf"), device=device)
    for i in range(0, n, chunk):
        d2 = torch.cdist(q, ref[i:i + chunk]).pow(2)
        # Exclude self-matches (distance 0) for query codes that fall in this chunk.
        in_chunk = (idx >= i) & (idx < i + chunk)
        rows = in_chunk.nonzero(as_tuple=True)[0]
        if len(rows):
            d2[rows, idx[rows] - i] = float("inf")
        nn_d2 = torch.minimum(nn_d2, d2.min(dim=1).values)

    nn_d2 = nn_d2.cpu().numpy()
    p5, p25, p50 = np.percentile(nn_d2, [5, 25, 50])
    print(f"Real-to-real squared-L2 NN distance (n={len(idx):,} sampled): "
          f"p5={p5:.3f}, p25={p25:.3f}, median={p50:.3f}")
    threshold = float(np.percentile(nn_d2, percentile))
    print(f"Using REJECT_THRESHOLD={threshold:.3f} (p{percentile} of real NN distances)")
    return threshold


def make_reject_fn(codes: torch.Tensor, device: str, threshold: float,
                   chunk: int = 50000):
    """Build `reject(query, cap) -> kept`, keeping codes whose nearest real code is
    farther than `threshold` (squared L2). `threshold <= 0` disables rejection.
    """
    if threshold <= 0:                                  # rejection disabled
        return lambda query, cap: query[:cap]

    ref = codes.to(device)

    def reject(query: torch.Tensor, cap: int) -> torch.Tensor:
        q = query.to(device)
        nn_d2 = torch.full((len(q),), float("inf"), device=device)
        for i in range(0, len(ref), chunk):
            d2 = torch.cdist(q, ref[i:i + chunk]).pow(2).min(dim=1).values
            nn_d2 = torch.minimum(nn_d2, d2)
        return q[nn_d2 > threshold][:cap].cpu()

    return reject


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_class(kde, n_target: int, model, diffusion, reject) -> torch.Tensor:
    """Decode `n_target` accepted fake slices from one class KDE into images."""
    out = torch.empty(n_target, 1, PX, PX)
    filled, stale = 0, 0
    pbar = tqdm(total=n_target, desc=f"generating {n_target}", unit="img")
    while filled < n_target:
        cand = torch.tensor(kde.resample(N_CANDIDATES).T, dtype=torch.float32)
        kept = reject(cand, cap=BATCH_CAP)
        if len(kept) == 0:                       # nothing passed the filter
            stale += 1
            if stale > 50:
                raise RuntimeError("Rejection threshold too strict (acceptance ~0) "
                                   "-- lower REJECT_THRESHOLD or set it to 0.")
            continue
        stale = 0
        imgs = diffusion.sample(model, n=len(kept), labels=kept.to(DEVICE),
                                cfg_scale=CFG_SCALE, angle_step=ANGLE_STEP,
                                noise_retention=NOISE_RETENTION)
        take = min(len(imgs), n_target - filled)
        out[filled:filled + take] = torch.from_numpy(imgs[:take])
        filled += take
        pbar.update(take)
    pbar.close()
    return out


def main() -> None:
    global DEVICE
    torch.backends.cudnn.benchmark = True
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    images_path = os.path.join(OUT_DIR, "fake_images.npy")
    labels_path = os.path.join(OUT_DIR, "fake_labels.npy")
    if os.path.exists(images_path) and not FORCE:
        sys.exit(f"{images_path} already exists -- set force: true in "
                 f"{os.path.basename(CONFIG_PATH)} to overwrite.")

    if DEVICE != "cpu" and not torch.cuda.is_available():
        DEVICE = "cpu"

    # --- Load codes, model, diffusion, KDEs, rejection filter --------------
    codes  = torch.from_numpy(np.load(os.path.join(CODES_DIR, "train_codes.npy")))
    labels = torch.from_numpy(np.load(os.path.join(CODES_DIR, "train_code_labels.npy")))
    model = load_model(CKPT, DEVICE)
    diffusion = load_diffusion(os.path.join(CODES_DIR, "norm_mean_std.npy"), DEVICE)
    neg_kde, pos_kde = fit_kdes(codes, labels, KDE_BANDWIDTH)

    threshold = REJECT_THRESHOLD
    if threshold is None:
        threshold = estimate_nn_threshold(codes, DEVICE, REJECT_PERCENTILE, NN_SAMPLE_SIZE)
    reject = make_reject_fn(codes, DEVICE, threshold)

    # --- Sanity-check the acceptance rate before the long run --------------
    acc = {name: len(reject(torch.tensor(kde.resample(2000).T, dtype=torch.float32), cap=2000)) / 2000
           for name, kde in [("PE-", neg_kde), ("PE+", pos_kde)]}
    print(f"Rejection acceptance (REJECT_THRESHOLD={threshold:.3f}): "
          f"PE- {acc['PE-']:.1%}, PE+ {acc['PE+']:.1%}")
    if max(acc.values()) < 0.01:
        sys.exit("Acceptance ~0 -- lower REJECT_THRESHOLD (or set it to 0 to disable rejection).")

    # --- Generate both classes ---------------------------------------------
    print("class_0 (PE-negative)")
    fake_neg = generate_class(neg_kde, N_NEG, model, diffusion, reject)
    print("class_1 (PE-positive)")
    fake_pos = generate_class(pos_kde, N_POS, model, diffusion, reject)

    images = torch.cat((fake_neg, fake_pos), dim=0).clip(-1000, 3000)   # neg first, then pos
    y = torch.cat((torch.zeros(N_NEG), torch.ones(N_POS)))

    if SHUFFLE:
        perm = torch.randperm(len(y))
        images, y = images[perm], y[perm]

    # --- Save --------------------------------------------------------------
    np.save(images_path, images.numpy())
    np.save(labels_path, y.numpy())
    print(f"\nSaved {len(images):,} fake images ({N_NEG:,} PE-, {N_POS:,} PE+)")
    print(f"  {images_path}   {tuple(images.shape)}")
    print(f"  {labels_path}")


if __name__ == "__main__":
    main()
