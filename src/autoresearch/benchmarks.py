"""Classic PSO benchmark / test functions.

All functions are *minimisation* targets with a known global optimum at 0
(or near 0).  They are widely used to evaluate optimiser performance and are
the subject of PSO Research Iteration 1.

References
----------
Molga, M. & Smutnicki, C. (2005). Test functions for optimization needs.
    Technical report.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Individual benchmark implementations
# ---------------------------------------------------------------------------


def sphere(x: np.ndarray) -> float:
    """Sphere function.

    f(x) = sum(x_i^2)

    Global minimum: f(0, ..., 0) = 0
    Typical search space: [-5.12, 5.12]^n
    """
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray, a: float = 10.0) -> float:
    """Rastrigin function.

    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))

    Global minimum: f(0, ..., 0) = 0
    Typical search space: [-5.12, 5.12]^n
    """
    n = len(x)
    return float(a * n + np.sum(x**2 - a * np.cos(2 * math.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock (banana) function.

    f(x) = sum_{i=0}^{n-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    Global minimum: f(1, ..., 1) = 0
    Typical search space: [-2.048, 2.048]^n
    """
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi) -> float:
    """Ackley function.

    Global minimum: f(0, ..., 0) = 0
    Typical search space: [-32.768, 32.768]^n
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return float(
        -a * math.exp(-b * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + a
        + math.e
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Map of function name → (callable, default_bounds, known_optimum_value)
BENCHMARKS: dict[str, tuple[Callable, list, float]] = {
    "sphere": (sphere, [-5.12, 5.12], 0.0),
    "rastrigin": (rastrigin, [-5.12, 5.12], 0.0),
    "rosenbrock": (rosenbrock, [-2.048, 2.048], 0.0),
    "ackley": (ackley, [-32.768, 32.768], 0.0),
}


def get_benchmark(name: str) -> tuple[Callable, list, float]:
    """Return ``(fn, default_bounds, optimum)`` for the named benchmark.

    Parameters
    ----------
    name:
        One of ``"sphere"``, ``"rastrigin"``, ``"rosenbrock"``, ``"ackley"``.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    if name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {list(BENCHMARKS.keys())}"
        )
    return BENCHMARKS[name]
