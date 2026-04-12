"""Spike encoding utilities for SNN rate and temporal coding.

Provides two encoding schemes:
  - Rate coding (Bernoulli): pixel intensity → Poisson spike probability over T steps.
  - TTFS coding (time-to-first-spike): pixel intensity → deterministic spike time.

Both return (T, N, *spatial) tensors of binary spikes compatible with snntorch.

Time-to-First-Spike (TTFS) encoding
-------------------------------------
Each pixel intensity p ∈ [0, 1] maps to a single spike at time t_fire:
    t_fire = floor((1 - p) × T)  for p ≥ threshold
    never fire                   for p < threshold

Bright pixels (p → 1) fire at t ≈ 0 (early).
Dark pixels (p → 0) fire at t ≈ T-1 (late) or not at all.
Result: each neuron fires at most once — maximal temporal information efficiency.

Unlike rate coding (many stochastic spikes), TTFS preserves exact firing times,
which is critical for the ANP's next-frame prediction objective.

References
----------
Thorpe et al. (1996): Rapid object recognition based on single spikes.
Gütig & Sompolinsky (2006): Time-to-first-spike coding in the retina.
"""

from typing import Optional

import torch


def rate_encode(
    x: torch.Tensor,
    timesteps: int,
    clamp: bool = True,
) -> torch.Tensor:
    """Bernoulli (Poisson) rate coding.

    Args:
        x:         (...) tensor with values in [0, 1] (or clamped if clamp=True).
        timesteps: Number of simulation timesteps T.
        clamp:     If True, clamp x to [0, 1] before encoding.

    Returns:
        (T, *x.shape) binary spike tensor — mean firing rate ≈ x.
    """
    if clamp:
        x = x.clamp(0.0, 1.0)
    return torch.bernoulli(x.unsqueeze(0).expand(timesteps, *x.shape))


def ttfs_encode(
    x: torch.Tensor,
    timesteps: int,
    threshold: Optional[float] = None,
    clamp: bool = True,
) -> torch.Tensor:
    """Time-to-first-spike encoding.

    Bright pixels fire early; dark pixels fire late or not at all.
    Each neuron fires at most once per sample.

    Args:
        x:          (...) tensor with values in [0, 1].
        timesteps:  Number of simulation timesteps T.
        threshold:  Minimum intensity to generate a spike.
                    Defaults to 1/timesteps (at least one step above zero).
        clamp:      If True, clamp x to [0, 1] before encoding.

    Returns:
        (T, *x.shape) binary spike tensor. At most one spike per neuron.
    """
    if clamp:
        x = x.clamp(0.0, 1.0)

    if threshold is None:
        threshold = 1.0 / timesteps  # smallest non-zero intensity level

    # Firing time: t_fire ∈ [0, T-1]
    # Bright (x=1) → t_fire=0; Dark (x→0) → t_fire→T-1
    t_fire = torch.floor((1.0 - x) * timesteps).long().clamp(0, timesteps - 1)

    # Only neurons above threshold fire
    fire_mask = x >= threshold  # (...) boolean

    # Build sparse one-hot spike tensor
    spikes = torch.zeros(timesteps, *x.shape, device=x.device, dtype=x.dtype)
    for t in range(timesteps):
        spikes[t] = ((t_fire == t) & fire_mask).float()

    return spikes  # (T, *x.shape)
