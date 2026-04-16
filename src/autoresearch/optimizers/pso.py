"""Particle Swarm Optimisation (PSO) — pure NumPy implementation.

Implements the standard global-best (gbest) PSO with optional inertia-weight
decay.  The algorithm is deliberately gradient-free and can optimise any
callable fitness function that accepts a 1-D NumPy array and returns a scalar.

References
----------
Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization.
    Proceedings of ICNN'95.
Shi, Y. & Eberhart, R. (1998). A Modified Particle Swarm Optimizer.
    Proceedings of IEEE WCCI'98.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PSOHistory:
    """Container for per-iteration PSO metrics."""

    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    swarm_std: List[float] = field(default_factory=list)


class PSO:
    """Global-best Particle Swarm Optimiser.

    Parameters
    ----------
    n_particles:
        Number of particles in the swarm.
    n_dims:
        Dimensionality of the search space.
    bounds:
        Array-like of shape (n_dims, 2) giving (lower, upper) bounds per
        dimension.  A single pair ``[lo, hi]`` is broadcast to all dims.
    w:
        Inertia weight (initial value when ``w_min`` is also set).
    c1:
        Cognitive (personal best) acceleration coefficient.
    c2:
        Social (global best) acceleration coefficient.
    w_min:
        If provided, inertia weight decays linearly from ``w`` to ``w_min``
        over the course of ``max_iter`` iterations.
    v_max_ratio:
        Maximum velocity as a fraction of the search-space width per dim.
        Set to ``None`` to disable velocity clamping.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_particles: int,
        n_dims: int,
        bounds: np.ndarray,
        w: float = 0.729,
        c1: float = 1.49445,
        c2: float = 1.49445,
        w_min: Optional[float] = None,
        v_max_ratio: Optional[float] = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.w = w
        self.w_init = w
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
        self.rng = np.random.default_rng(seed)

        # Normalise bounds → shape (n_dims, 2)
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim == 1:
            bounds = np.broadcast_to(bounds, (n_dims, 2)).copy()
        self.bounds = bounds  # (n_dims, 2)

        self.lo = bounds[:, 0]  # (n_dims,)
        self.hi = bounds[:, 1]  # (n_dims,)
        self.span = self.hi - self.lo  # (n_dims,)

        # Velocity limits
        if v_max_ratio is not None:
            self.v_max = v_max_ratio * self.span
        else:
            self.v_max = None

        # Swarm state (initialised in _init_swarm)
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.personal_best_pos: Optional[np.ndarray] = None
        self.personal_best_fit: Optional[np.ndarray] = None
        self.global_best_pos: Optional[np.ndarray] = None
        self.global_best_fit: float = float("inf")

    # ------------------------------------------------------------------
    # Swarm initialisation
    # ------------------------------------------------------------------

    def _init_swarm(self) -> None:
        """Initialise particle positions and velocities randomly."""
        # Positions: uniform in [lo, hi]
        self.positions = (
            self.rng.uniform(0, 1, size=(self.n_particles, self.n_dims)) * self.span
            + self.lo
        )

        # Velocities: ±0.5 × span (common heuristic)
        self.velocities = self.rng.uniform(
            -0.5 * self.span,
            0.5 * self.span,
            size=(self.n_particles, self.n_dims),
        )

        self.personal_best_pos = self.positions.copy()
        self.personal_best_fit = np.full(self.n_particles, float("inf"))
        self.global_best_pos = self.positions[0].copy()
        self.global_best_fit = float("inf")

    # ------------------------------------------------------------------
    # Core update step
    # ------------------------------------------------------------------

    def _step(self, iteration: int, max_iter: int) -> None:
        """Perform one PSO velocity/position update."""
        r1 = self.rng.uniform(0, 1, size=(self.n_particles, self.n_dims))
        r2 = self.rng.uniform(0, 1, size=(self.n_particles, self.n_dims))

        # Inertia weight (linear decay if w_min is set)
        if self.w_min is not None:
            w = self.w_init - (self.w_init - self.w_min) * iteration / max(max_iter - 1, 1)
        else:
            w = self.w

        # Velocity update
        cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social = self.c2 * r2 * (self.global_best_pos - self.positions)
        self.velocities = w * self.velocities + cognitive + social

        # Velocity clamping
        if self.v_max is not None:
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

        # Position update
        self.positions = self.positions + self.velocities

        # Position clamping (reflect at boundary)
        self.positions = np.clip(self.positions, self.lo, self.hi)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        max_iter: int = 100,
        convergence_tol: float = 1e-8,
        verbose: bool = True,
        log_frequency: int = 10,
    ) -> Tuple[np.ndarray, float, PSOHistory]:
        """Run the PSO optimisation loop.

        Parameters
        ----------
        fitness_fn:
            Callable ``f(x: np.ndarray) -> float`` to minimise.  ``x`` has
            shape ``(n_dims,)``.
        max_iter:
            Maximum number of iterations.
        convergence_tol:
            Stop early when the improvement in global-best fitness across 10
            consecutive iterations falls below this threshold.
        verbose:
            Whether to emit log messages.
        log_frequency:
            How often (in iterations) to log progress.

        Returns
        -------
        best_pos:
            Position of the global-best particle.
        best_fit:
            Fitness value at ``best_pos``.
        history:
            Per-iteration metrics.
        """
        self._init_swarm()
        history = PSOHistory()

        prev_best = float("inf")
        stagnant_count = 0
        stagnation_window = 10

        for iteration in range(max_iter):
            # Evaluate all particles
            fitnesses = np.array(
                [fitness_fn(self.positions[i]) for i in range(self.n_particles)]
            )

            # Update personal bests
            improved = fitnesses < self.personal_best_fit
            self.personal_best_pos[improved] = self.positions[improved]
            self.personal_best_fit[improved] = fitnesses[improved]

            # Update global best
            best_idx = int(np.argmin(fitnesses))
            if fitnesses[best_idx] < self.global_best_fit:
                self.global_best_fit = float(fitnesses[best_idx])
                self.global_best_pos = self.positions[best_idx].copy()

            # Record history
            history.best_fitness.append(self.global_best_fit)
            history.mean_fitness.append(float(np.mean(fitnesses)))
            history.swarm_std.append(float(np.std(self.positions)))

            if verbose and (iteration % log_frequency == 0 or iteration == max_iter - 1):
                log.info(
                    "Iter %4d/%d | best=%.6e | mean=%.6e | swarm_std=%.4f",
                    iteration + 1,
                    max_iter,
                    self.global_best_fit,
                    history.mean_fitness[-1],
                    history.swarm_std[-1],
                )

            # Early stopping on stagnation
            if abs(prev_best - self.global_best_fit) < convergence_tol:
                stagnant_count += 1
                if stagnant_count >= stagnation_window:
                    log.info(
                        "Converged at iteration %d (tol=%.2e)", iteration + 1, convergence_tol
                    )
                    break
            else:
                stagnant_count = 0
            prev_best = self.global_best_fit

            # Update velocities/positions
            self._step(iteration, max_iter)

        return self.global_best_pos, self.global_best_fit, history
