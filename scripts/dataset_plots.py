"""Rich diagnostic plots for a competition dataset.

Loads a saved .npz and produces:
  plots/<tag>_lhc.png            — LHC corner
  plots/<tag>_spectra_random.png — 200 random curves, plain overlay
  plots/<tag>_spectra_envelope.png — min/max + 5/95 percentile + median + Planck fid
  plots/<tag>_spectra_by_<param>.png — colored by one parameter, for each of
                                        H0, ln10^{10}A_s, omega_cdm, n_s
  plots/<tag>_spectra_dispersion.png — fractional std vs ell
  plots/<tag>_chi2_hist.png      — if test set: full chi^2_true(i,j) + CDF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmbemu.box import PARAM_NAMES, PLANCK_FIDUCIAL
from cmbemu.generate import generate_spectra
from cmbemu.io import load_dataset
from cmbemu.likelihood import chi2_matrix
from cmbemu.plotting import (
    plot_chi2_distribution,
    plot_lhc,
    plot_spectra_colored_by,
    plot_spectra_dispersion,
    plot_spectra_envelope,
    plot_spectra_random,
)


def _planck_cls(lmax_cmb: int, lmax_pp: int) -> dict:
    params = np.array([[PLANCK_FIDUCIAL[n] for n in PARAM_NAMES]])
    return generate_spectra(params, lmax_cmb=lmax_cmb, lmax_pp=lmax_pp,
                            progress_every=0, log=lambda s: None)


def produce(tag: str, do_chi2: bool = False) -> None:
    path = ROOT / "data" / f"{tag}.npz"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    d = load_dataset(path)
    N = d["params"].shape[0]
    lmax_cmb, lmax_pp = d["lmax_cmb"], d["lmax_pp"]
    cls = {k: d[k] for k in ("tt", "te", "ee", "pp")}
    params = d["params"]

    plots = ROOT / "plots"
    print(f"\n=== {tag}  (N={N}, cosmo_model={d['cosmo_model']}) ===")

    print("[lhc]")
    plot_lhc(params, plots / f"{tag}_lhc.png", title=f"LHC — {tag}")

    print("[spectra] random")
    plot_spectra_random(cls, plots / f"{tag}_spectra_random.png",
                        title=f"Spectra — {tag}", n_curves=200)

    print("[spectra] envelope + Planck fiducial")
    fid = _planck_cls(lmax_cmb, lmax_pp)
    plot_spectra_envelope(cls, plots / f"{tag}_spectra_envelope.png",
                          title=f"Spectra envelope — {tag}", fiducial_cls=fid)

    for cparam in ("H0", "ln10^{10}A_s", "omega_cdm", "n_s"):
        safe = cparam.replace("^", "").replace("{", "").replace("}", "").replace("\\", "_")
        print(f"[spectra] colored by {cparam}")
        plot_spectra_colored_by(cls, params, cparam,
                                plots / f"{tag}_spectra_by_{safe}.png",
                                title=f"Spectra colored by {cparam} — {tag}",
                                n_curves=300)

    print("[spectra] fractional dispersion")
    plot_spectra_dispersion(cls, plots / f"{tag}_spectra_dispersion.png",
                            title=f"Fractional spectra dispersion — {tag}",
                            fiducial_cls=fid)

    if do_chi2:
        print("[chi2] computing and plotting")
        chi2 = chi2_matrix(d, d, lmax_cmb=lmax_cmb, lmax_pp=lmax_pp)
        plot_chi2_distribution(chi2, plots / f"{tag}_chi2_hist.png",
                               title=fr"$\chi^2_{{\rm true}}(i,j)$ — {tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+",
                        default=["train_small", "test_small"])
    args = parser.parse_args()
    for tag in args.tags:
        # Only compute chi^2 matrix for test-like sets (NxN, modest N)
        do_chi2 = tag.startswith("test")
        produce(tag, do_chi2=do_chi2)
    print("\nall done.")


if __name__ == "__main__":
    main()
