"""Build a small (5k train + 500 test) version of the competition dataset.

Produces:
  data/train_small.npz
  data/test_small.npz
  plots/lhc_train.png, plots/lhc_test.png
  plots/spectra_train.png, plots/spectra_test.png
"""
from __future__ import annotations

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


N_TRAIN = 5000
N_TEST = 500
SEED_TRAIN = 202604
SEED_TEST = 202605


def build(n: int, seed: int, out_path: Path, tag: str) -> None:
    print(f"\n=== building {tag}: n={n}, seed={seed} ===")
    print(f"[lhc] sampling")
    params = sample_lhc(n, seed=seed)

    print(f"[lhc] plotting")
    plot_lhc(params, ROOT / "plots" / f"lhc_{tag}.png", title=f"LHC / {tag}")

    print(f"[gen] generating {n} spectra")
    t0 = time.perf_counter()
    cls = generate_spectra(params, lmax_cmb=LMAX_CMB, lmax_pp=LMAX_PP,
                           progress_every=500)
    dt = time.perf_counter() - t0
    print(f"[gen] done in {dt:.1f}s  ({dt/n*1000:.1f} ms/call)")

    print(f"[io]  saving {out_path}")
    try:
        csz_version = _pkg_version("classy_sz")
    except Exception:
        csz_version = "unknown"
    save_dataset(out_path, params, cls,
                 metadata={"classy_sz_version": csz_version, "seed": seed,
                           "cosmo_model": COSMO_MODEL})

    print(f"[plot] spectra subset")
    plot_spectra_random(cls, ROOT / "plots" / f"spectra_{tag}.png",
                 title=f"Spectra / {tag}")


def main() -> None:
    build(N_TRAIN, SEED_TRAIN, ROOT / "data" / "train_small.npz", "train")
    build(N_TEST, SEED_TEST, ROOT / "data" / "test_small.npz", "test")
    print("\nall done.")


if __name__ == "__main__":
    main()
