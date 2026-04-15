"""NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for SimCLR.

Reference:
    Chen et al. (2020). A Simple Framework for Contrastive Learning of
    Visual Representations. https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """NT-Xent contrastive loss.

    Given N samples with two augmented views z1 and z2 (each of shape
    (N, D)), treats (z1_i, z2_i) as a positive pair and all other 2N-2
    samples as negatives.

    Both z1 and z2 are assumed to be L2-normalised (unit vectors).

    Args:
        z1:          First set of projected embeddings, shape (N, D).
        z2:          Second set of projected embeddings, shape (N, D).
        temperature: Softmax temperature τ.  Smaller values concentrate
                     the distribution (sharper), larger values soften it.

    Returns:
        Scalar NT-Xent loss averaged over 2N terms.
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    # Cosine similarity matrix (L2-norm already applied externally)
    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Mask out self-similarity (diagonal) to exclude trivial pairs
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive pair labels:
    #   z1[i] (row i) is positive with z2[i] at index i+N
    #   z2[i] (row i+N) is positive with z1[i] at index i
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])

    return F.cross_entropy(sim, labels)
