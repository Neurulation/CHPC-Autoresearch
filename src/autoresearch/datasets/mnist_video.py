"""MNIST-Video dataset — stacked same-class MNIST frames.

A simple temporal dataset built on top of standard MNIST: each sample is a
sequence of ``seq_len`` images all belonging to the same digit class, sampled
uniformly at random from the training or test split. The sequence label is the
shared digit class.

This is the simplest possible temporal extension of MNIST — no new data downloads
required. It validates the temporal dataloading pipeline before moving to NMNIST
and real event-camera data.

Usage
-----
    from autoresearch.datasets.mnist_video import MNISTVideo
    ds = MNISTVideo(root="./data", train=True, seq_len=8, download=True)
    frames, label = ds[0]   # frames: (seq_len, 1, 28, 28),  label: int
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTVideo(Dataset):
    """Sequential MNIST: groups of same-class frames returned as short videos.

    Args:
        root:      Directory for MNIST data cache.
        train:     Use training split if True, else test split.
        seq_len:   Number of frames per sequence.
        transform: Optional per-frame transform (applied to each frame independently).
        download:  Download MNIST if not already cached.
        seed:      Random seed for reproducible index construction.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        seq_len: int = 8,
        transform=None,
        download: bool = True,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len

        if transform is None:
            transform = transforms.ToTensor()

        base = datasets.MNIST(root=root, train=train, transform=transform, download=download)

        # Group indices by class label
        class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(base):
            class_indices[int(label)].append(idx)

        self._base = base
        self._class_indices = {k: v for k, v in class_indices.items()}
        self._classes = sorted(self._class_indices.keys())
        self._rng = random.Random(seed)

        # Pre-build index: list of (class, [frame_idx_0, ..., frame_idx_{seq_len-1}])
        # Length = number of base samples (each base image anchors one sequence)
        self._sequences: List[Tuple[int, List[int]]] = []
        for cls, idxs in self._class_indices.items():
            for anchor in idxs:
                # Sample seq_len indices from the same class (with replacement if needed)
                pool = idxs if len(idxs) >= seq_len else idxs
                chosen = self._rng.choices(pool, k=seq_len)
                self._sequences.append((cls, chosen))

        # Shuffle so digit classes are interleaved
        self._rng.shuffle(self._sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return a sequence of frames and the shared class label.

        Returns:
            frames: (seq_len, C, H, W) float tensor.
            label:  int digit class.
        """
        label, frame_indices = self._sequences[idx]
        frames = torch.stack([self._base[i][0] for i in frame_indices])
        return frames, label
