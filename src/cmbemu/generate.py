"""Generate CMB spectra with classy_szfast for a batch of LHC points."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .box import COSMO_MODEL, LMAX_CMB, LMAX_PP, PARAM_NAMES, params_array_to_dict


def _get_classy_sz(cosmo_model: int = COSMO_MODEL):
    """Instantiate a classy_sz Class and initialize classy_szfast.

    cosmo_model is set *before* initialize_classy_szfast (required — passing
    it per get_cmb_cls call does nothing). Defaults to COSMO_MODEL from box.py.
    """
    from classy_sz import Class as Class_sz
    csz = Class_sz()
    csz.set({"cosmo_model": cosmo_model})
    csz.initialize_classy_szfast()
    return csz


def _truncate(out: dict, lmax_cmb: int, lmax_pp: int) -> dict[str, np.ndarray]:
    return {
        "tt": np.asarray(out["tt"][: lmax_cmb + 1], dtype=np.float32),
        "te": np.asarray(out["te"][: lmax_cmb + 1], dtype=np.float32),
        "ee": np.asarray(out["ee"][: lmax_cmb + 1], dtype=np.float32),
        "pp": np.asarray(out["pp"][: lmax_pp + 1], dtype=np.float32),
    }


def generate_spectra(
    params: np.ndarray,
    lmax_cmb: int = LMAX_CMB,
    lmax_pp: int = LMAX_PP,
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> dict[str, np.ndarray]:
    """Run classy_szfast for each row of params (shape (N, 6)).

    Returns a dict of float32 arrays: tt/te/ee of shape (N, lmax_cmb+1) and
    pp of shape (N, lmax_pp+1). Raises on any failure so we don't silently
    drop samples.
    """
    if params.ndim != 2 or params.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"params must be (N, 6), got {params.shape}")

    csz = _get_classy_sz()

    N = params.shape[0]
    tt = np.empty((N, lmax_cmb + 1), dtype=np.float32)
    te = np.empty((N, lmax_cmb + 1), dtype=np.float32)
    ee = np.empty((N, lmax_cmb + 1), dtype=np.float32)
    pp = np.empty((N, lmax_pp + 1), dtype=np.float32)

    # Warm up
    csz.get_cmb_cls(params_values_dict=params_array_to_dict(params[0]))

    t0 = time.perf_counter()
    for i in range(N):
        d = params_array_to_dict(params[i])
        out = csz.get_cmb_cls(params_values_dict=d)
        trunc = _truncate(out, lmax_cmb, lmax_pp)
        tt[i] = trunc["tt"]
        te[i] = trunc["te"]
        ee[i] = trunc["ee"]
        pp[i] = trunc["pp"]
        if progress_every and (i + 1) % progress_every == 0:
            dt = time.perf_counter() - t0
            per = dt / (i + 1)
            eta = per * (N - i - 1)
            log(f"  gen {i+1}/{N}  ({per*1e3:.1f} ms/call, eta {eta:.0f}s)")

    return {"tt": tt, "te": te, "ee": ee, "pp": pp}
