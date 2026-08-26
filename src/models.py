from typing import Optional, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: Any, new: Any) -> Any:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    @staticmethod
    def reset_parameters(ema_model: nn.Module, model: nn.Module) -> None:
        ema_model.load_state_dict(model.state_dict())


class ClassifierWrap(nn.Module):
    def __init__(self, core, emb_dim, n_out):
        super().__init__()

        self.core = core
        self.dense = nn.Linear(emb_dim, n_out)

    def forward(self, x):
        x = self.core(x)
        x = F.silu(x)
        x = self.dense(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
            self,
            px: int, f_unet: List[int], f_classif: List[int], blocks: int, emb_dim: int,
            device: Union[torch.device, str]
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.unet = UnetConditional(f_unet, blocks, time_dim=emb_dim, c_in=1, c_out=1).to(device)
        self.classif = Classifier(px, f_classif, blocks, n_out=emb_dim, c_in=1).to(device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, support=None) -> torch.Tensor:
        if support is None:
            labels = None
        else:
            labels = self.classif(support)
        return self.unet(x, t, labels)


class SelfAttentionGate(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, f0: int, filters: int, depth: int) -> None:
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.depth = depth
        layers = []
        self.res_conv = nn.Conv2d(f0, filters, kernel_size=(1, 1)) if f0 != filters else None
        for i in range(depth):
            f = filters if i > 0 else f0
            layers.append(nn.GroupNorm(1, f))
            layers.append(nn.Conv2d(f, filters, kernel_size=(3, 3), padding=1))
            layers.append(nn.SiLU())
            layers.append(nn.Conv2d(filters, filters, kernel_size=(3, 3), padding=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.res_conv is None else self.res_conv(x)
        for i, l in enumerate(self.layers):
            x = l(x)
            if (i + 1) % 4 == 0:
                x += res
                res = x
        return x


class EmbGate(nn.Module):
    def __init__(self, out_channels: int, emb_dim: int = 256) -> None:
        super().__init__()

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])


class UnetConditional(nn.Module):
    def __init__(
            self,
            filters: List[int], block_size: int, c_in: int = 3, c_out: int = 3, time_dim: int = 256,
            device: Union[torch.device, str] = "cuda"
    ) -> None:
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = nn.Conv2d(c_in, filters[0], kernel_size=3, padding=1)
        self.outc = nn.Conv2d(filters[0], c_out, kernel_size=1)

        embs = []
        layers_down_res = []
        layer_down_attn = []
        layers_up_res = []
        layers_up_attn = []

        for i, f in enumerate(filters[:-1]):
            f0 = filters[i - 1] if i > 0 else filters[0]
            layers_down_res.append(ResidualBlock(f0, f, block_size))
            layer_down_attn.append(SelfAttentionGate(f))
            embs.append(EmbGate(f, time_dim))

        self.lb = ResidualBlock(filters[-2], filters[-1], block_size)
        embs.append(EmbGate(filters[-1], time_dim))

        for i, f in enumerate(reversed(filters[:-1])):
            f0 = f + filters[::-1][i]
            layers_up_res.append(ResidualBlock(f0, f, block_size))
            layers_up_attn.append(SelfAttentionGate(f))
            embs.append(EmbGate(f, time_dim))

        self.layers_u = nn.ModuleList(layers_up_res)
        self.layers_d = nn.ModuleList(layers_down_res)
        self.layers_u1 = nn.ModuleList(layers_up_attn)
        self.layers_d1 = nn.ModuleList(layer_down_attn)
        self.layers_e = nn.ModuleList(embs)

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # `t` is reassigned twice below: raw diffusion angle -> sinusoidal positional encoding
        # -> encoding + semantic label embedding `y` (the combined conditioning vector used by
        # every EmbGate in the down/up path).
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += y

        x = self.inc(x)

        skips = []

        i = 0  # embedding layer count

        for layer_res, layer_attn in zip(self.layers_d, self.layers_d1):
            x = layer_res(x)
            x = layer_attn(x)
            x += self.layers_e[i](t, x)
            i += 1
            skips.append(x)
            x = nn.MaxPool2d(2)(x)

        x = self.lb(x)
        x += self.layers_e[i](t, x)
        i += 1

        for layer_res, layer_attn in zip(self.layers_u, self.layers_u1):
            x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(x)
            popped = skips.pop()
            x = torch.cat([popped, x], 1)
            x = layer_res(x)
            x = layer_attn(x)
            x += self.layers_e[i](t, x)
            i += 1

        output = self.outc(x)
        return output


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


class UNet(nn.Module):
    def __init__(
            self,
            filters: List[int], block_size: int, c_in: int = 3, c_out: int = 3,
            device: Union[torch.device, str] = "cuda"
    ) -> None:
        super().__init__()
        self.device = device

        self.inc = nn.Conv2d(c_in, filters[0], kernel_size=3, padding=1)
        self.outc = nn.Conv2d(filters[0], c_out, kernel_size=1)

        layers_down_res = []
        layers_down_attn = []
        layers_up_res = []
        layers_up_attn = []

        for i, f in enumerate(filters[:-1]):
            f0 = filters[i - 1] if i > 0 else filters[0]
            layers_down_res.append(ResidualBlock(f0, f, block_size))
            layers_down_attn.append(SelfAttentionGate(f))

        self.lb = ResidualBlock(filters[-2], filters[-1], block_size)

        for i, f in enumerate(reversed(filters[:-1])):
            f0 = f + filters[::-1][i]
            layers_up_res.append(ResidualBlock(f0, f, block_size))
            layers_up_attn.append(SelfAttentionGate(f))

        self.layers_u = nn.ModuleList(layers_up_res)
        self.layers_d = nn.ModuleList(layers_down_res)
        self.layers_u1 = nn.ModuleList(layers_up_attn)
        self.layers_d1 = nn.ModuleList(layers_down_attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.inc(x)

        skips = []

        for layer_res, layer_attn in zip(self.layers_d, self.layers_d1):
            x = layer_res(x)
            x = layer_attn(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)

        x = self.lb(x)

        for layer_res, layer_attn in zip(self.layers_u, self.layers_u1):
            x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(x)
            popped = skips.pop()
            x = torch.cat([popped, x], 1)
            x = layer_res(x)
            x = layer_attn(x)

        output = self.outc(x)
        return output
