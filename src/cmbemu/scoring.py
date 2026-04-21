"""Pairwise chi^2 precision score (MAE on Delta chi^2) and combined
precision/speed score.

The primary score is the MAE of the total chi^2 (CMB + PP). Alongside it we
compute two natural sub-scores whose emulators can be composed independently:

- CMB block  (2x2 TT/TE/EE Wishart)
- PP block   (1x1 independent Wishart)

Per-spectrum raw-Cl MAE (TT-only, TE-only, EE-only, PP-only) is available as
a diagnostic in api.get_score, but is not a physically valid likelihood.
"""
from __future__ import annotations

import numpy as np


def _mae_offdiag(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected square matrix, got {a.shape}")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    N = a.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return {
        "mae": float(diff[mask].mean()),
        "median_abs_diff": float(np.median(diff[mask])),
        "max_abs_diff": float(diff[mask].max()),
        "n_pairs": int(mask.sum()),
    }


def chi2_mae(
    chi2_true: np.ndarray,
    chi2_emu: np.ndarray,
    include_diagonal: bool = False,
) -> dict:
    """Mean absolute error of chi^2 over all off-diagonal pairs.

    S_prec = mean_{i != j} |chi^2_emu(i,j) - chi^2_true(i,j)|
    """
    if chi2_true.shape != chi2_emu.shape:
        raise ValueError(
            f"shape mismatch: true {chi2_true.shape}, emu {chi2_emu.shape}"
        )
    if chi2_true.ndim != 2 or chi2_true.shape[0] != chi2_true.shape[1]:
        raise ValueError(f"expected square matrix, got {chi2_true.shape}")

    diff = np.abs(chi2_emu.astype(np.float64) - chi2_true.astype(np.float64))
    N = chi2_true.shape[0]
    if include_diagonal:
        mae = float(diff.mean())
        n = N * N
        med = float(np.median(diff))
        mx = float(diff.max())
    else:
        mask = ~np.eye(N, dtype=bool)
        mae = float(diff[mask].mean())
        n = N * (N - 1)
        med = float(np.median(diff[mask]))
        mx = float(diff[mask].max())

    return {
        "mae": mae,
        "n_pairs": n,
        "max_abs_diff": mx,
        "median_abs_diff": med,
    }


def chi2_mae_blocks(
    split_true: dict[str, np.ndarray],
    split_emu: dict[str, np.ndarray],
) -> dict[str, dict]:
    """MAE on the total, CMB, and PP chi^2 blocks.

    Inputs come from likelihood.chi2_matrix_split().
    """
    return {
        "total": _mae_offdiag(split_true["total"], split_emu["total"]),
        "cmb":   _mae_offdiag(split_true["cmb"],   split_emu["cmb"]),
        "pp":    _mae_offdiag(split_true["pp"],    split_emu["pp"]),
    }


def combined_score(mae: float, t_cpu_ms: float, alpha: float = 1.0) -> float:
    """S = log10(mae) + alpha * log10(t_cpu_ms)."""
    return float(np.log10(mae) + alpha * np.log10(t_cpu_ms))
