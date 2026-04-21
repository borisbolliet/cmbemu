"""Deeper look at the chi^2_true(i, j) distribution on test_small."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmbemu.io import load_dataset
from cmbemu.likelihood import chi2_matrix


def main() -> None:
    test = load_dataset(ROOT / "data" / "test_small.npz")
    N = test["params"].shape[0]
    print(f"N = {N}")

    chi2 = chi2_matrix(test, test,
                       lmax_cmb=test["lmax_cmb"], lmax_pp=test["lmax_pp"])
    print(f"chi2_matrix shape {chi2.shape}")

    off = chi2[~np.eye(N, dtype=bool)].astype(np.float64)
    off = off[np.isfinite(off)]
    print(f"off-diagonal pairs: {len(off)}")
    print(f"  min     = {off.min():.3e}")
    print(f"  max     = {off.max():.3e}")
    print(f"  mean    = {off.mean():.3e}")
    print(f"  median  = {np.median(off):.3e}")
    qs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99]
    for q in qs:
        print(f"  q{q:<6g} = {np.quantile(off, q):.3e}")

    print("\nsurvivors at various cuts:")
    total = len(off)
    for cut in (40, 100, 1000, 10000, 1e5, 1e6, 1e7, 1e8, 1e9):
        n = int((off < cut).sum())
        print(f"  cut={cut:>10g} : {n:>8d}  ({100*n/total:.2f}%)")


if __name__ == "__main__":
    main()
