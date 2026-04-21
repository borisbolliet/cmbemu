"""Run the scorer end-to-end on the small dataset.

1. Loads data/test_small.npz.
2. Computes chi^2_true(i, j) for all test pairs.
3. Plots the chi^2 distribution.
4. Runs a dummy 'constant-Planck' emulator through the scorer.
5. Reports chi^2_MSE, n_pairs surviving, and combined score.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmbemu.box import LMAX_CMB, LMAX_PP, PARAM_NAMES, PLANCK_FIDUCIAL
from cmbemu.io import load_dataset
from cmbemu.likelihood import chi2_matrix
from cmbemu.plotting import plot_chi2_distribution, plot_score_diagnostic
from cmbemu.scoring import chi2_mae, combined_score


def main() -> None:
    test = load_dataset(ROOT / "data" / "test_small.npz")
    N = test["params"].shape[0]
    print(f"[load] N_test = {N}, lmax_cmb={test['lmax_cmb']}, lmax_pp={test['lmax_pp']}")

    # chi2_true: all test pairs
    print("[chi2] computing chi2_true(i, j), NxN...")
    t0 = time.perf_counter()
    chi2_true = chi2_matrix(test, test,
                            lmax_cmb=test["lmax_cmb"], lmax_pp=test["lmax_pp"])
    print(f"[chi2] done in {time.perf_counter() - t0:.2f}s, shape {chi2_true.shape}")
    print(f"       diagonal (should be ~0): mean {np.abs(np.diag(chi2_true)).mean():.2e}, "
          f"max {np.abs(np.diag(chi2_true)).max():.2e}")

    # Distribution
    print("[plot] chi2_true distribution")
    survivors = plot_chi2_distribution(
        chi2_true,
        ROOT / "plots" / "chi2_true_hist.png",
        title="chi^2_true(i,j) on test_small",
        cuts=(40.0, 100.0, 1000.0, 10000.0),
    )
    total = N * (N - 1)
    for cut, n_surv in sorted(survivors.items()):
        print(f"       chi2 < {cut:>7g} : {n_surv:>8d} / {total}  ({100*n_surv/total:.2f}%)")

    # Dummy emulator: always outputs Planck fiducial regardless of params.
    # Build cls_emu by replicating the first test sample's classy_sz-computed
    # spectrum at the Planck fiducial... but our test LHC doesn't include
    # the exact Planck point. Simpler: generate the Planck spectrum fresh.
    print("\n[emu] constant-Planck dummy emulator")
    from cmbemu.generate import generate_spectra
    planck_arr = np.array([[PLANCK_FIDUCIAL[n] for n in PARAM_NAMES]])
    planck_cls = generate_spectra(planck_arr, lmax_cmb=test["lmax_cmb"],
                                  lmax_pp=test["lmax_pp"], progress_every=0)

    emu_cls = {
        k: np.repeat(planck_cls[k], N, axis=0) for k in ("tt", "te", "ee", "pp")
    }
    print(f"[emu] emu shape for tt: {emu_cls['tt'].shape}")

    print("[chi2] computing chi2_emu(i, j)...")
    t0 = time.perf_counter()
    chi2_emu = chi2_matrix(test, emu_cls,
                           lmax_cmb=test["lmax_cmb"], lmax_pp=test["lmax_pp"])
    print(f"[chi2] done in {time.perf_counter() - t0:.2f}s")

    # Diagnostic: emu vs true scatter
    plot_score_diagnostic(
        chi2_true, chi2_emu,
        ROOT / "plots" / "score_diag_constant_planck.png",
        title="dummy: constant-Planck emulator",
    )

    print("\n[score] MAE on Delta chi^2 (off-diagonal only):")
    res = chi2_mae(chi2_true, chi2_emu)
    print(f"  n_pairs        = {res['n_pairs']}")
    print(f"  mean |Delta|   = {res['mae']:.3e}")
    print(f"  median |Delta| = {res['median_abs_diff']:.3e}")
    print(f"  max |Delta|    = {res['max_abs_diff']:.3e}")

    # Placeholder timing; will be measured in the real scorer.
    t_ms = 10.0
    S = combined_score(res["mae"], t_ms, alpha=1.0)
    print(f"\n[combined] with placeholder t_cpu = {t_ms} ms:  S = {S:.3f}")


if __name__ == "__main__":
    main()
