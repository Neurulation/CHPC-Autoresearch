"""MNIST-compatible augmentations for contrastive self-supervised learning.

Provides a two-view augmentation pipeline adapted for MNIST (grayscale 28×28,
pixel values in [0, 1]).  Each call returns two independently augmented views
of the same image for use as a positive pair in NT-Xent / SimCLR training.

Pipeline (applied independently twice per image):
  1. Random crop with padding   — spatial invariance
  2. Random horizontal flip     — shape invariance (low probability; digits are
                                   mostly asymmetric but some benefit)
  3. Random brightness jitter   — illumination invariance
  4. Random contrast jitter     — illumination invariance
  5. Gaussian noise             — robustness to sensor noise
  6. Random erasing             — occlusion robustness
"""

import random

import torch
import torchvision.transforms.functional as TF


class MNISTContrastiveAugmentation:
    """Two-view augmentation pipeline for MNIST contrastive learning.

    Args:
        crop_size:     Output image size after random crop (default 28).
        padding:       Padding added on each side before cropping (default 4).
        flip_p:        Probability of horizontal flip (default 0.2).
        brightness:    Max absolute brightness factor delta (default 0.4).
        contrast:      Max absolute contrast factor delta (default 0.4).
        noise_std:     Standard deviation of additive Gaussian noise (default 0.1).
        erasing_p:     Probability of applying random erasing (default 0.3).
        erasing_scale: (min, max) fraction of image area to erase (default (0.02, 0.15)).
    """

    def __init__(
        self,
        crop_size: int = 28,
        padding: int = 4,
        flip_p: float = 0.2,
        brightness: float = 0.4,
        contrast: float = 0.4,
        noise_std: float = 0.1,
        erasing_p: float = 0.3,
        erasing_scale: tuple = (0.02, 0.15),
    ):
        self.crop_size = crop_size
        self.padding = padding
        self.flip_p = flip_p
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.erasing_p = erasing_p
        self.erasing_scale = erasing_scale

    def __call__(self, x: torch.Tensor) -> tuple:
        """Apply two independent augmentations to an image.

        Args:
            x: Image tensor of shape (C, H, W) with values in [0, 1].

        Returns:
            Tuple (view1, view2) each of shape (C, H, W).
        """
        return self._augment(x), self._augment(x)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a single random augmentation chain to x."""
        # 1. Random crop with reflection padding
        padded = TF.pad(x, self.padding, padding_mode="reflect")
        max_i = padded.size(1) - self.crop_size
        max_j = padded.size(2) - self.crop_size
        i = random.randint(0, max_i)
        j = random.randint(0, max_j)
        x = TF.crop(padded, i, j, self.crop_size, self.crop_size)

        # 2. Random horizontal flip
        if random.random() < self.flip_p:
            x = TF.hflip(x)

        # 3. Brightness jitter
        if self.brightness > 0 and random.random() < 0.8:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            x = (x * factor).clamp(0.0, 1.0)

        # 4. Contrast jitter
        if self.contrast > 0 and random.random() < 0.8:
            mean = x.mean()
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            x = ((x - mean) * factor + mean).clamp(0.0, 1.0)

        # 5. Gaussian noise
        if self.noise_std > 0 and random.random() < 0.5:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)

        # 6. Random erasing
        if random.random() < self.erasing_p:
            # scale = fraction of image area to erase; sqrt gives side length
            scale = random.uniform(*self.erasing_scale)
            side = max(1, int(self.crop_size * scale ** 0.5))
            i_e = random.randint(0, self.crop_size - side)
            j_e = random.randint(0, self.crop_size - side)
            x = x.clone()
            x[:, i_e : i_e + side, j_e : j_e + side] = 0.0

        return x
