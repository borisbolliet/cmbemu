"""Latin hypercube sampling in the competition box."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .box import PARAM_NAMES, box_bounds


def sample_lhc(n: int, seed: int) -> np.ndarray:
    """Scrambled Latin hypercube of shape (n, 6) inside the competition box.

    Uses scipy.stats.qmc.LatinHypercube with fixed seed for reproducibility.
    """
    lo, hi = box_bounds()
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    u = sampler.random(n)
    return lo + u * (hi - lo)
