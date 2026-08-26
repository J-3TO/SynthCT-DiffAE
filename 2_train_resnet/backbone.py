"""Backbone classifiers used by 2p1_train_pe_classifiers.py.
"""
from typing import List, Union
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.models import ResidualBlock, SelfAttentionGate  # noqa: E402


class Classifier(nn.Module):
    def __init__(
            self,
            px: int, filters: List[int], block_size: int, c_in: int = 3, n_out: int = 16,
            device: Union[torch.device, str] = "cuda"
    ) -> None:
        super().__init__()
        self.device = device

        self.inc = nn.Conv2d(c_in, filters[0], kernel_size=3, padding=1)
        self.outc = nn.Linear(int(filters[-1] * (px / 2 ** (len(filters))) ** 2), n_out)

        layers_down_res = []
        llayers_down_attn = []

        for i, f in enumerate(filters):
            f0 = filters[i - 1] if i > 0 else filters[0]
            layers_down_res.append(ResidualBlock(f0, f, block_size))
            llayers_down_attn.append(SelfAttentionGate(f))

        self.layers_d = nn.ModuleList(layers_down_res)
        self.layers_d1 = nn.ModuleList(llayers_down_attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)

        for layer_res, layer_attn in zip(self.layers_d, self.layers_d1):
            x = layer_res(x)
            x = layer_attn(x)
            x = nn.MaxPool2d(2)(x)

        x = nn.Flatten()(x)
        output = self.outc(x)
        return output


class ClassifierWrap(nn.Module):
    """Encoder embedding -> SiLU -> linear classification head."""

    def __init__(self, core: nn.Module, emb_dim: int, n_out: int) -> None:
        super().__init__()
        self.core = core
        self.dense = nn.Linear(emb_dim, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(F.silu(self.core(x)))


class ReplicateToRGB(nn.Module):
    """Repeats a 1-channel input to 3 channels so an RGB-pretrained backbone can be reused
    unmodified """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def build_resnet50(c_in: int = 1, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    resnet = torchvision.models.resnet50(weights=weights)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    if c_in not in (1, 3):
        raise ValueError(f"build_resnet50 supports c_in in (1, 3), got {c_in}")
    return nn.Sequential(ReplicateToRGB(), resnet)


def build_classifier(backbone: str = "custom", c_in: int = 1, num_classes: int = 2,
                     pretrained: bool = False,
                     px: int = 256, emb_dim: int = 512,
                     filters=(32, 64, 128, 256, 512), block_size: int = 3) -> nn.Module:
    if backbone == "custom":
        core = Classifier(px, list(filters), block_size, c_in=c_in, n_out=emb_dim, device="cpu")
        return ClassifierWrap(core, emb_dim, num_classes)
    if backbone == "resnet50":
        return build_resnet50(c_in=c_in, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown backbone: {backbone!r} (expected 'custom' or 'resnet50')")
