"""Trivial reference emulators that conform to the competition interface.

The contract is very small: an emulator is any object with a
    predict(params: dict[str, float]) -> dict[str, np.ndarray]
method. The returned dict carries keys "tt", "te", "ee", "pp" with
shapes (LMAX_CMB + 1,) for the first three and (LMAX_PP + 1,) for "pp".
"""
from __future__ import annotations

import numpy as np

from .box import LMAX_CMB, LMAX_PP, PARAM_NAMES, PLANCK_FIDUCIAL


class ConstantPlanck:
    """Emulator that returns the Planck fiducial spectrum regardless of input.

    Useful as a leaderboard lower bound: any emulator that can't beat this is
    ignoring its inputs entirely.
    """

    name = "ConstantPlanck"

    def __init__(self, lmax_cmb: int = LMAX_CMB, lmax_pp: int = LMAX_PP):
        # Lazy-compute the fiducial spectrum on first call to avoid pulling
        # classy_sz at import time.
        self._lmax_cmb = lmax_cmb
        self._lmax_pp = lmax_pp
        self._cls: dict[str, np.ndarray] | None = None

    def _ensure_fid(self) -> None:
        if self._cls is not None:
            return
        from .generate import generate_spectra
        params = np.array([[PLANCK_FIDUCIAL[n] for n in PARAM_NAMES]])
        fid = generate_spectra(params, lmax_cmb=self._lmax_cmb,
                               lmax_pp=self._lmax_pp, progress_every=0,
                               log=lambda s: None)
        self._cls = {k: fid[k][0].copy() for k in ("tt", "te", "ee", "pp")}

    def predict(self, params: dict) -> dict[str, np.ndarray]:
        self._ensure_fid()
        return {k: self._cls[k].copy() for k in ("tt", "te", "ee", "pp")}


class NearestNeighbour:
    """1-NN lookup on a training set, using Euclidean distance in the
    normalized parameter box (each axis mapped to [0, 1]).

    Not a real emulator — just a leaderboard baseline that uses the training
    data with zero interpolation.
    """

    name = "NearestNeighbour"

    def __init__(self, train: dict):
        """train is a loaded dataset dict (from load_train / load_dataset)."""
        from .box import box_bounds
        self._train = train
        self._params = np.asarray(train["params"], dtype=np.float64)
        lo, hi = box_bounds()
        self._lo = lo
        self._span = hi - lo
        self._u = (self._params - lo) / self._span  # (N, d)

    def predict(self, params: dict) -> dict[str, np.ndarray]:
        theta = np.array([params[n] for n in PARAM_NAMES], dtype=np.float64)
        u = (theta - self._lo) / self._span
        d2 = np.sum((self._u - u) ** 2, axis=1)
        j = int(np.argmin(d2))
        return {
            "tt": self._train["tt"][j].copy(),
            "te": self._train["te"][j].copy(),
            "ee": self._train["ee"][j].copy(),
            "pp": self._train["pp"][j].copy(),
        }
