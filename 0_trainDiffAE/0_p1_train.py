import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import yaml
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
from torch import optim
import torch
from functools import partial

from src.training_tools import train, log_reco_results
from src.utils import ImageLoader, load_data, get_codes
from src.diffusion import Diffusion
from src.models import Autoencoder

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / 'train_config.yml'
RESULT_DIR = REPO_ROOT / 'results'

# Checkpoint fields in train_config.yml are stored relative to the repo root.
_CHECKPOINT_FIELDS = (
    'small_model_checkpoint', 'big_model_pretrained_checkpoint', 'big_model_fine_tuned_checkpoint'
)
_DATASET_PATH_FIELDS = ('ds_path', 'ds_meta')


def load_config(path: Path) -> SimpleNamespace:
    """
    Load train_config.yml into a mutable, dot-accessible config object 
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for field in _CHECKPOINT_FIELDS:
        cfg[field] = REPO_ROOT / cfg[field]
    for field in _DATASET_PATH_FIELDS:
        if cfg.get(field) is not None:
            cfg[field] = Path(cfg[field])
    return SimpleNamespace(**cfg)


CONFIG = load_config(CONFIG_PATH)
DEVICE = CONFIG.device

if CONFIG.device_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.device_id)


def load_data_for_training(
        begin: Optional[int] = None, end: Optional[int] = None, org_res: Optional[int] = None
) -> (torch.Tensor, torch.Tensor):
    begin = begin or CONFIG.ds_id_first
    end = end or CONFIG.ds_id_last
    org_res = org_res or CONFIG.img_res

    ds_256 = load_data(CONFIG.ds_path, CONFIG.ds_meta, begin, end, org_res, 256)[0]
    ds_128 = F.interpolate(ds_256, size=128, mode='nearest')
    return ds_128, ds_256


def init_models(images: torch.Tensor) -> (Autoencoder, Autoencoder, Diffusion):
    model_128 = Autoencoder(128, [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], 3, 512, DEVICE)
    model_256 = Autoencoder(256, [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], 3, 512, DEVICE)

    diffusion = Diffusion(img_size=128, device=DEVICE, lat_size=512, blurring=CONFIG.use_blur)
    diffusion.std = images.std(dim=(0, 2, 3))
    diffusion.mean = images.mean(dim=(0, 2, 3))

    return model_128, model_256, diffusion


def train_small_model(
        diffusion: Diffusion, small_model: Autoencoder, dataloader: ImageLoader, vis_ex_imgs: torch.Tensor
) -> None:
    print('Training small model...')
    optimizer = optim.AdamW(small_model.parameters(), lr=CONFIG.lr_128)
    train(
        diffusion, small_model, dataloader, optimizer, CONFIG.small_model_epochs, CONFIG,
        use_ema=True, is_supervised=False,
        logging_fc=partial(log_reco_results, diffusion, 'small_model', vis_ex_imgs, config=CONFIG)
    )
    state_dict_128_path = CONFIG.small_model_checkpoint
    torch.save(small_model.state_dict(), state_dict_128_path)
    print(f'Small model saved to {state_dict_128_path}')


def prepare_labels_dataloader(
        diffusion: Diffusion, small_model: Autoencoder, ds_128: torch.Tensor, ds256: torch.Tensor
) -> ImageLoader:
    labels_set = get_codes(small_model, diffusion, ds_128, CONFIG.batch_128)
    rand_order = torch.randperm(len(ds256))
    ds_shuffled = ds256[rand_order]
    labels_shuffled = labels_set[rand_order]
    dataloader = ImageLoader(ds_shuffled, labels_shuffled, batch_size=CONFIG.batch_256)
    return dataloader


def pretrain_big_model(
        diffusion: Diffusion, big_model: Autoencoder, small_model: Autoencoder,
        ds_128: torch.Tensor, ds_256: torch.Tensor, vis_ex_imgs: torch.Tensor
) -> None:
    print('Label extraction...')
    supervised_dataloader = prepare_labels_dataloader(diffusion, small_model, ds_128, ds_256)

    print('Pretraining bigger model...')

    print('Fitting DPM decoder...')
    optimizer = optim.AdamW(big_model.unet.parameters(), lr=CONFIG.lr_256)
    diffusion.img_size = 256
    train(
        diffusion, big_model, supervised_dataloader, optimizer, epochs=1, config=CONFIG,
        encoder=False, use_ema=True, is_supervised=True,
        logging_fc=partial(log_reco_results, diffusion, 'big_fixed', vis_ex_imgs, config=CONFIG)
    )

    print('Fitting semantic encoder...')
    optimizer = optim.AdamW(big_model.classif.parameters(), lr=CONFIG.lr_256)
    train(
        diffusion, big_model, supervised_dataloader, optimizer, epochs=1, config=CONFIG,
        decoder=False, use_ema=False, is_supervised=True, logging_fc=None
    )
    state_dict_pretrained_path = CONFIG.big_model_pretrained_checkpoint
    torch.save(big_model.state_dict(), state_dict_pretrained_path)
    print(f'Pretrained big model saved to {state_dict_pretrained_path}')


def fine_tune_big_model(
        diffusion: Diffusion, big_model: Autoencoder, ds_256: torch.Tensor, vis_ex_imgs: torch.Tensor
) -> None:
    print('Fine-tuning big model...')
    diffusion.img_size = 256
    optimizer = optim.AdamW(big_model.parameters(), lr=CONFIG.lr_256)
    dataloader_256 = ImageLoader(ds_256, batch_size=CONFIG.batch_256)

    train(
        diffusion, big_model, dataloader_256, optimizer, epochs=CONFIG.final_training_epochs,
        config=CONFIG, use_ema=True, is_supervised=False,
        logging_fc=partial(log_reco_results, diffusion, 'big_freed', vis_ex_imgs, config=CONFIG)
    )
    state_dict_fine_tuned_path = CONFIG.big_model_fine_tuned_checkpoint
    torch.save(big_model.state_dict(), state_dict_fine_tuned_path)
    print(f'Fine-tuned big model saved to {state_dict_fine_tuned_path}')


def extract_codes(model: Autoencoder, diffusion: Diffusion, ds: torch.Tensor):
    codes = get_codes(model, diffusion, ds, CONFIG.batch_256)
    torch.save(codes, RESULT_DIR / 'training_codes.pt')


def main():
    if CONFIG.ds_path is None or CONFIG.ds_meta is None:
        raise SystemExit(f"Set ds_path/ds_meta in {CONFIG_PATH} before running.")

    if CONFIG.seed is not None:
        torch.manual_seed(CONFIG.seed)
        np.random.seed(CONFIG.seed)

    ds_128, ds_256 = load_data_for_training()
    model_small, model_big, diffusion = init_models(ds_128)

    rand_order = torch.randperm(len(ds_256))[:4]
    progress_visualisation_examples = {"small": ds_128[rand_order], "big": ds_256[rand_order]}

    if CONFIG.train_small:
        dataloader_128 = ImageLoader(ds_128, batch_size=CONFIG.batch_128)
        train_small_model(diffusion, model_small, dataloader_128, progress_visualisation_examples["small"])

    if CONFIG.pretrain_big:
        state_dict_128_path = CONFIG.small_model_checkpoint
        assert state_dict_128_path.exists(), 'Trained small model checkpoint does not exist.'
        model_small.load_state_dict(torch.load(state_dict_128_path))
        pretrain_big_model(diffusion, model_big, model_small, ds_128, ds_256, progress_visualisation_examples["big"])

    if CONFIG.fine_tune_big:
        state_dict_pretrained_path = CONFIG.big_model_pretrained_checkpoint
        assert state_dict_pretrained_path.exists(), 'Pretrained big model checkpoint does not exist.'
        model_big.load_state_dict(torch.load(state_dict_pretrained_path))
        fine_tune_big_model(diffusion, model_big, ds_256, progress_visualisation_examples["big"])

        print('Training finished, extracting codes for later use...')
        extract_codes(model_big, diffusion, ds_256)


if __name__ == "__main__":
    main()
