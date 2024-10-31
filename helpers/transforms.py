"""
Author: Carlo Alberto Barbano (carlo.barbano@unito.it)
Date: 22/04/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from PIL import Image

class ToTensor(nn.Module):
    def __init__(self):
        self.to_tensor = torchvision.transforms.PILToTensor()
    def forward(self, x: Image):
        return self.to_tensor(x)

class TotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()


class Normalize(nn.Module):
    # From https://github.com/hamidkazemi22/CLIPInversion/blob/main/invert.py
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Jitter(nn.Module):
    # From https://github.com/hamidkazemi22/CLIPInversion/blob/main/helpers/augmentations.py#L21
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


class ColorJitter(nn.Module):
    # From https://github.com/hamidkazemi22/CLIPInversion/blob/main/helpers/augmentations.py#L57
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1., device="cpu"):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.device = device
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,), device=self.device) - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,), device=self.device) - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std


class Scale(nn.Module):
    def __init__(self, size, mode='bicubic'):
        super(Scale, self).__init__()
        self.mode = mode
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode=self.mode)


class Repeat(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.repeat(self.n, 1, 1, 1)


class NViewTransform:
    """Create N augmented views of the same image"""
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        return torch.cat([self.transform(x) for _ in range(self.n)], dim=0)
