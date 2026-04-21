"""Verify the vectorized chi^2 against a straightforward per-ell-loop reference.

Compares three quantities for correctness:
  1. Diagonal must be ~0 (theory == data).
  2. Off-diagonal: matmul chi^2 vs brute-force per-ell loop with np.linalg.inv.
  3. Symmetry check: chi^2(i, i) == 0 is an identity, but chi^2(i, j) != chi^2(j, i)
     in general (it's not a metric) — we don't require symmetry, only
     self-consistency between two implementations.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmbemu.box import F_SKY
from cmbemu.io import load_dataset
from cmbemu.likelihood import chi2_matrix


def chi2_bruteforce(data_cls, theory_cls, i: int, j: int,
                    lmax_cmb: int, lmax_pp: int, f_sky: float = F_SKY) -> float:
    """Reference: explicit loop over ell with numpy 2x2 inverse per multipole."""
    total = 0.0
    # CMB block
    for ell in range(2, lmax_cmb + 1):
        C_d = np.array([
            [data_cls["tt"][i, ell], data_cls["te"][i, ell]],
            [data_cls["te"][i, ell], data_cls["ee"][i, ell]],
        ], dtype=np.float64)
        C_t = np.array([
            [theory_cls["tt"][j, ell], theory_cls["te"][j, ell]],
            [theory_cls["te"][j, ell], theory_cls["ee"][j, ell]],
        ], dtype=np.float64)
        C_t_inv = np.linalg.inv(C_t)
        trace = np.trace(C_d @ C_t_inv)
        sign_t, logdet_t = np.linalg.slogdet(C_t)
        sign_d, logdet_d = np.linalg.slogdet(C_d)
        if sign_t <= 0 or sign_d <= 0:
            return float("inf")
        bracket = trace + logdet_t - logdet_d - 2.0
        total += (2 * ell + 1) * f_sky * bracket

    # PP block
    for ell in range(2, lmax_pp + 1):
        pp_d = float(data_cls["pp"][i, ell])
        pp_t = float(theory_cls["pp"][j, ell])
        bracket = pp_d / pp_t + np.log(pp_t) - np.log(pp_d) - 1.0
        total += (2 * ell + 1) * f_sky * bracket

    return total


def main() -> None:
    test = load_dataset(ROOT / "data" / "test_small.npz")
    lmax_cmb = test["lmax_cmb"]
    lmax_pp = test["lmax_pp"]
    N = test["params"].shape[0]
    print(f"loaded N={N}, lmax_cmb={lmax_cmb}, lmax_pp={lmax_pp}")

    # Full matmul chi^2
    t0 = time.perf_counter()
    chi2_vec = chi2_matrix(test, test, lmax_cmb=lmax_cmb, lmax_pp=lmax_pp)
    print(f"matmul chi2_matrix: {time.perf_counter() - t0:.2f}s, shape {chi2_vec.shape}")

    # Brute-force spot checks
    rng = np.random.default_rng(1)
    pairs = [(rng.integers(N), rng.integers(N)) for _ in range(6)]
    # Force a few diagonal entries too
    pairs += [(0, 0), (N // 2, N // 2), (N - 1, N - 1)]

    print("\npair        bruteforce               matmul                   |rel_err|")
    print("-" * 80)
    for (i, j) in pairs:
        t0 = time.perf_counter()
        chi2_bf = chi2_bruteforce(test, test, i, j, lmax_cmb, lmax_pp)
        chi2_v = float(chi2_vec[i, j])
        if chi2_bf == 0.0 and chi2_v == 0.0:
            rel = 0.0
        elif abs(chi2_bf) < 1e-6:
            rel = abs(chi2_v - chi2_bf)  # absolute when bf is ~0
        else:
            rel = abs(chi2_v - chi2_bf) / abs(chi2_bf)
        tag = "(diag)" if i == j else ""
        print(f"({i:>3d},{j:>3d}) {tag:<7s} {chi2_bf: .10e}   {chi2_v: .10e}   {rel:.2e}")


if __name__ == "__main__":
    main()
