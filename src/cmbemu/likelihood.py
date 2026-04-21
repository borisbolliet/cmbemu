"""CV-limited Gaussian (Wishart) log-likelihood for CMB TT/TE/EE + lensing phi-phi.

All heavy work is vectorized over (data_point, theory_point, multipole) via
BLAS matmul, so computing the full N_data x N_theory chi^2 matrix is O(seconds)
on CPU for N ~ O(10^3).

Shapes by convention:
  cls["tt"], cls["te"], cls["ee"]: (N, lmax_cmb + 1)  (ell = 0, 1, ..., lmax_cmb)
  cls["pp"]:                       (N, lmax_pp  + 1)

Primary entry points:
  - chi2_matrix(data, theory, ...)         total NxN chi^2
  - chi2_matrix_split(data, theory, ...)   returns ("cmb", "pp", "total")
  - chi2_diag(data, theory, ...)           diagonal only
"""
from __future__ import annotations

import numpy as np

from .box import F_SKY, LMAX_CMB, LMAX_PP


def _cmb_prep(cls: dict[str, np.ndarray], lmax_cmb: int) -> dict[str, np.ndarray]:
    tt = cls["tt"][:, 2 : lmax_cmb + 1].astype(np.float64)
    te = cls["te"][:, 2 : lmax_cmb + 1].astype(np.float64)
    ee = cls["ee"][:, 2 : lmax_cmb + 1].astype(np.float64)
    det = tt * ee - te * te
    return {"tt": tt, "te": te, "ee": ee, "det": det, "logdet": np.log(det)}


def _pp_prep(cls: dict[str, np.ndarray], lmax_pp: int) -> dict[str, np.ndarray]:
    pp = cls["pp"][:, 2 : lmax_pp + 1].astype(np.float64)
    return {"pp": pp, "logpp": np.log(pp)}


def _cmb_block(data_cls, theory_cls, lmax_cmb: int, f_sky: float) -> np.ndarray:
    ell = np.arange(2, lmax_cmb + 1, dtype=np.float64)
    w = (2.0 * ell + 1.0) * f_sky
    d = _cmb_prep(data_cls, lmax_cmb)
    t = _cmb_prep(theory_cls, lmax_cmb)

    inv_det_t = 1.0 / t["det"]
    A_t = t["ee"] * inv_det_t * w
    B_t = t["te"] * inv_det_t * w
    C_t = t["tt"] * inv_det_t * w
    trace_term = d["tt"] @ A_t.T - 2.0 * d["te"] @ B_t.T + d["ee"] @ C_t.T

    logdet_t_w = (t["logdet"] * w).sum(axis=1)
    logdet_d_w = (d["logdet"] * w).sum(axis=1)
    w_tot = w.sum()

    return trace_term + logdet_t_w[None, :] - logdet_d_w[:, None] - 2.0 * w_tot


def _pp_block(data_cls, theory_cls, lmax_pp: int, f_sky: float) -> np.ndarray:
    ell = np.arange(2, lmax_pp + 1, dtype=np.float64)
    w = (2.0 * ell + 1.0) * f_sky
    dp = _pp_prep(data_cls, lmax_pp)
    tp = _pp_prep(theory_cls, lmax_pp)

    inv_pp_t = 1.0 / tp["pp"]
    Q_t = inv_pp_t * w
    ratio_term = dp["pp"] @ Q_t.T

    logpp_t_w = (tp["logpp"] * w).sum(axis=1)
    logpp_d_w = (dp["logpp"] * w).sum(axis=1)
    w_tot = w.sum()

    return ratio_term + logpp_t_w[None, :] - logpp_d_w[:, None] - w_tot


def chi2_matrix_split(
    data_cls: dict[str, np.ndarray],
    theory_cls: dict[str, np.ndarray],
    lmax_cmb: int = LMAX_CMB,
    lmax_pp: int = LMAX_PP,
    f_sky: float = F_SKY,
) -> dict[str, np.ndarray]:
    """Return {"cmb": ..., "pp": ..., "total": cmb+pp}, each (N_data, N_theory).

    The CMB block is the 2x2 TT/TE/EE Wishart; PP is the independent 1x1
    Wishart. These are the natural composability units (see scoring.md).
    """
    cmb = _cmb_block(data_cls, theory_cls, lmax_cmb, f_sky)
    pp = _pp_block(data_cls, theory_cls, lmax_pp, f_sky)
    return {"cmb": cmb, "pp": pp, "total": cmb + pp}


def chi2_matrix(
    data_cls: dict[str, np.ndarray],
    theory_cls: dict[str, np.ndarray],
    lmax_cmb: int = LMAX_CMB,
    lmax_pp: int = LMAX_PP,
    f_sky: float = F_SKY,
) -> np.ndarray:
    """Return chi2[i, j] = -2 log L(theory_j | data_i), shape (N_data, N_theory).

    Decomposes each multipole sum into BLAS matmuls so the full (i, j, ell)
    tensor is never materialized. Uses float64 internally for stability of the
    2x2 determinants.
    """
    return chi2_matrix_split(data_cls, theory_cls, lmax_cmb, lmax_pp, f_sky)["total"]


def chi2_diag(
    data_cls: dict[str, np.ndarray],
    theory_cls: dict[str, np.ndarray],
    lmax_cmb: int = LMAX_CMB,
    lmax_pp: int = LMAX_PP,
    f_sky: float = F_SKY,
) -> np.ndarray:
    """Return chi2[i] = -2 log L(theory_i | data_i), shape (N,).

    Useful as a cheap diagnostic (self-chi2 along diagonal).
    """
    N = data_cls["tt"].shape[0]
    if theory_cls["tt"].shape[0] != N:
        raise ValueError("data and theory must have matching N for chi2_diag")
    ell_cmb = np.arange(2, lmax_cmb + 1, dtype=np.float64)
    ell_pp = np.arange(2, lmax_pp + 1, dtype=np.float64)
    w_cmb = (2.0 * ell_cmb + 1.0) * f_sky
    w_pp = (2.0 * ell_pp + 1.0) * f_sky

    d = _cmb_prep(data_cls, lmax_cmb)
    t = _cmb_prep(theory_cls, lmax_cmb)
    dp = _pp_prep(data_cls, lmax_pp)
    tp = _pp_prep(theory_cls, lmax_pp)

    trace = (
        d["tt"] * t["ee"] - 2.0 * d["te"] * t["te"] + d["ee"] * t["tt"]
    ) / t["det"]
    bracket_cmb = trace + (t["logdet"] - d["logdet"]) - 2.0
    chi2_cmb = (w_cmb * bracket_cmb).sum(axis=1)

    bracket_pp = dp["pp"] / tp["pp"] + (tp["logpp"] - dp["logpp"]) - 1.0
    chi2_pp = (w_pp * bracket_pp).sum(axis=1)

    return chi2_cmb + chi2_pp
