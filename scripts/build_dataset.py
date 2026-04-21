"""Build the competition dataset (full size or small).

Usage:
    python scripts/build_dataset.py --size full    # 50k train + 5k test
    python scripts/build_dataset.py --size small   #  5k train + 500 test
"""
from __future__ import annotations

import argparse
import sys
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmbemu.box import COSMO_MODEL, LMAX_CMB, LMAX_PP
from cmbemu.generate import generate_spectra
from cmbemu.io import save_dataset
from cmbemu.lhc import sample_lhc
from cmbemu.plotting import plot_lhc, plot_spectra_random


SIZES = {
    "full":  {"train": (50_000, 202604), "test": (5_000, 202605),
              "train_path": "train.npz",       "test_path": "test.npz"},
    "small": {"train": ( 5_000, 202604), "test": (  500, 202605),
              "train_path": "train_small.npz", "test_path": "test_small.npz"},
}


def build(tag: str, n: int, seed: int, out_path: Path) -> None:
    print(f"\n=== building {tag}: n={n}, seed={seed} ===")
    print(f"[lhc] sampling")
    params = sample_lhc(n, seed=seed)

    print(f"[lhc] plotting (dpi=300 corner)")
    plot_lhc(params, ROOT / "plots" / f"{out_path.stem}_lhc.png",
             title=f"LHC — {out_path.stem}")

    print(f"[gen] generating {n} spectra")
    t0 = time.perf_counter()
    cls = generate_spectra(params, lmax_cmb=LMAX_CMB, lmax_pp=LMAX_PP,
                           progress_every=max(1000, n // 50))
    dt = time.perf_counter() - t0
    print(f"[gen] done in {dt:.1f}s  ({dt/n*1000:.2f} ms/call)")

    print(f"[io]  saving {out_path}")
    try:
        csz_version = _pkg_version("classy_sz")
    except Exception:
        csz_version = "unknown"
    save_dataset(out_path, params, cls,
                 metadata={"classy_sz_version": csz_version,
                           "seed": seed,
                           "cosmo_model": COSMO_MODEL})

    print(f"[plot] 200-random spectra")
    plot_spectra_random(cls, ROOT / "plots" / f"{out_path.stem}_spectra_random.png",
                        title=f"Spectra — {out_path.stem}", n_curves=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="full", choices=list(SIZES.keys()))
    args = parser.parse_args()

    cfg = SIZES[args.size]
    for tag in ("train", "test"):
        n, seed = cfg[tag]
        out_path = ROOT / "data" / cfg[f"{tag}_path"]
        build(tag, n, seed, out_path)
    print("\nall done.")


if __name__ == "__main__":
    main()
