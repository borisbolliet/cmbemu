"""Verify the 2x2 TT/TE/EE covariance is positive definite at every (sample, ell)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmbemu.io import load_dataset


def main() -> None:
    for tag in ("train_small", "test_small"):
        d = load_dataset(ROOT / "data" / f"{tag}.npz")
        tt = d["tt"][:, 2:].astype(np.float64)
        te = d["te"][:, 2:].astype(np.float64)
        ee = d["ee"][:, 2:].astype(np.float64)
        pp = d["pp"][:, 2:].astype(np.float64)

        det = tt * ee - te * te
        print(f"\n=== {tag}  (N={d['params'].shape[0]}, lmax_cmb={d['lmax_cmb']}) ===")
        print(f"  TT   min {tt.min():.3e}, max {tt.max():.3e}, any<=0 : {(tt<=0).any()}")
        print(f"  EE   min {ee.min():.3e}, max {ee.max():.3e}, any<=0 : {(ee<=0).any()}")
        print(f"  det  min {det.min():.3e}, any<=0 : {(det<=0).any()}  (neg count: {(det<=0).sum()})")
        print(f"  pp   min {pp.min():.3e}, any<=0 : {(pp<=0).any()}")

        # Cauchy-Schwarz ratio: |TE|^2 / (TT * EE) should be < 1
        with np.errstate(invalid="ignore"):
            ratio = te**2 / (tt * ee)
        finite = np.isfinite(ratio)
        print(f"  |TE|^2/(TT*EE) max (should be <1): {ratio[finite].max():.6f}")
        if (ratio[finite] >= 1.0).any():
            bad = np.where((ratio >= 1.0) & finite)
            print(f"    VIOLATIONS at {len(bad[0])} points; first: sample={bad[0][0]}, ell={bad[1][0]+2}")


if __name__ == "__main__":
    main()
