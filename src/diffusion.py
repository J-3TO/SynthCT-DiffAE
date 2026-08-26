from typing import Union, Optional

import numpy as np
import torch
import torch.fft
from tqdm import tqdm
import torchvision

from src.models import Autoencoder

class Diffusion:
    def __init__(
            self,
            img_size: int = 256, device: Union[torch.device, str] = "cuda",
            a0: int = 3, a1: int = 180, sr0: int = 0.2, lat_size: int = 256, blurring: bool = True
    ) -> None:

        # sr0/sr1: the cosine signal-rate schedule's endpoints 
        self.sr0 = torch.tensor(sr0, dtype=torch.float)
        self.sr1 = torch.tensor(0.95, dtype=torch.float)
        self.img_size = img_size
        self.device = device
        self.a0 = a0
        self.a1 = a1
        self.lat_size = lat_size
        self.blurring = blurring

        self.mean = None
        self.std = None

    def t_of_a(self, a: torch.Tensor) -> torch.Tensor:
        """Map an integer diffusion "angle" a in [a0, a1] to a [0, 1]-like fraction (1 = noise, 0 = clean)."""
        return 1 - (a - self.a0) / (self.a1 - self.a0)

    def get_frequency_mask(
            self,
            t: torch.Tensor, min_scale: float = 0.001, sino: bool = False, img_dim: int = None
    ) -> torch.Tensor:
        t = (0.1 / (0.115 - 0.1 * t)) ** 2 - 0.756
        img_dim = self.img_size // 2 if img_dim is None else img_dim
        time = t ** 2 / 2

        freq = np.pi * torch.cat(
            [torch.linspace(0, img_dim - 1, img_dim), torch.linspace(img_dim, 0, img_dim)]) / img_dim
        labda = freq[None, None, :] ** 2 if sino else freq[None, None, None, :] ** 2 + freq[None, None, :, None] ** 2
        time = time[:, None, None] if sino else time[:, None, None, None]

        scaling = torch.exp(-labda.to(self.device) * time) * (1 - min_scale)
        scaling = scaling + min_scale
        return scaling

    def blur(self, imgs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fftn(imgs, dim=(-1, -2))
        m = self.get_frequency_mask(t)
        filtered = f * m
        blurred = torch.fft.ifftn(filtered, dim=(-1, -2))
        return blurred

    def deblur(self, imgs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fftn(imgs, dim=(-1, -2))
        m = self.get_frequency_mask(t)
        filtered = f / m
        deblurred = torch.fft.ifftn(filtered, dim=(-1, -2))
        return deblurred

    def blur_sino(self, sin: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fftn(sin.squeeze(1), dim=-1)
        m = self.get_frequency_mask(t, img_dim=sin.shape[-1] // 2, sino=True)
        sin_b = torch.real(torch.fft.ifftn(f * m, dim=-1))
        return sin_b.unsqueeze(1)

    def noise_images(self, x: torch.Tensor, a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        nr, sr = self.diffusion_schedule(a)
        noise = torch.randn_like(x)
        return sr * x + nr * noise, noise

    def sample_angles(self, n: int) -> torch.Tensor:
        return torch.randint(low=self.a0, high=self.a1 + 1, size=(n,))

    @staticmethod
    def int_to_timesteps(t: torch.Tensor, n: int) -> torch.Tensor:
        return torch.ones((n,)) * t

    def diffusion_schedule(self, angles: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        diffusion_times = self.t_of_a(angles)
        diffusion_times = diffusion_times.unsqueeze(-1)
        diffusion_times = diffusion_times.unsqueeze(-1)
        diffusion_times = diffusion_times.unsqueeze(-1)
        start_angle = torch.acos(self.sr1)
        end_angle = torch.acos(self.sr0)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates.to(self.device), signal_rates.to(self.device)

    def sample(
            self,
            model: Autoencoder, n: int, images: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
            cfg_scale: float = 3, angle_step: int = 1, noise_retention: float = 0.5,
            init_angle: int = 3, skip_last_angles: int = 0
    ) -> np.ndarray:
        """Reverse-diffusion sampling. 
        """
        model.eval()
        img = None
        with torch.no_grad():
            labels = model.classif(images.to(self.device)) if labels is None else labels.to(self.device)
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for a in tqdm(range(init_angle, self.a1 + 1 - skip_last_angles, angle_step), position=0):
                t = self.t_of_a(torch.tensor(a)) * torch.ones(n).to(self.device)
                nr, sr = self.diffusion_schedule(self.int_to_timesteps(torch.tensor(a), n))
                if img is not None:
                    if self.blurring:
                        img = self.blur(img, t).real
                    x = sr * img + nr * (noise_retention ** 0.5 * predicted_noise + (1 - noise_retention) ** 0.5 * torch.randn_like(img))
                # Small Gaussian jitter (/200) on the conditioning label 
                predicted_noise = model.unet(x, a * torch.ones(n).to(self.device),
                                             labels + torch.randn(n, self.lat_size).to(self.device) / 200)
                if cfg_scale != 0:
                    # The "unconditional" branch CFG extrapolates away from
                    uncond_predicted_noise = model.unet(x, a * torch.ones(n).to(self.device),
                                                        torch.randn(n, self.lat_size).to(self.device) / 30)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                img = (x - predicted_noise * nr) / sr
                if self.blurring:
                    img = self.deblur(img, t)
        img = torchvision.transforms.Normalize(-self.mean, std=1)(
            torchvision.transforms.Normalize(0, self.std ** -1)(img.real))
        model.train()
        return img.cpu().numpy()
