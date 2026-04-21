"""High-level convenience API for the competition.

Exposed through ``cmbemu`` as:
    cmbemu.generate_data(n, seed, save_to=...)
    cmbemu.get_accuracy_score(emulator, ...)     # precision only
    cmbemu.get_time_score(emulator, ...)         # timing only
    cmbemu.get_score(emulator, ...)              # both + combined_S
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .benchmark import benchmark_emulator
from .box import (
    COSMO_MODEL,
    LMAX_CMB,
    LMAX_PP,
    params_array_to_dict,
)
from .io import save_dataset
from .lhc import sample_lhc
from .likelihood import chi2_matrix_split
from .scoring import chi2_mae_blocks, combined_score


class EmulatorProtocol(Protocol):
    """What a submitted emulator must provide."""
    def predict(self, params: dict) -> dict[str, np.ndarray]: ...


def generate_data(
    n: int,
    seed: int,
    save_to: str | Path | None = None,
    lmax_cmb: int = LMAX_CMB,
    lmax_pp: int = LMAX_PP,
    progress_every: int = 500,
) -> dict:
    """Generate N spectra on a Latin-hypercube draw of the competition box.

    Uses classy_szfast with ``cosmo_model=6`` (ede-v2). Returns a dict with
    keys ``params, tt, te, ee, pp`` plus metadata. Optionally writes a .npz.

    Parameters
    ----------
    n : number of LHC points
    seed : RNG seed for the LHC sampler
    save_to : if given, also save the dataset as .npz at that path
    """
    from importlib.metadata import version as _pkg_version
    from .generate import generate_spectra

    params = sample_lhc(n, seed=seed)
    cls = generate_spectra(params, lmax_cmb=lmax_cmb, lmax_pp=lmax_pp,
                           progress_every=progress_every)

    try:
        csz_version = _pkg_version("classy_sz")
    except Exception:
        csz_version = "unknown"

    out = {
        "params": params,
        **cls,
        "seed": seed,
        "lmax_cmb": lmax_cmb,
        "lmax_pp": lmax_pp,
        "cosmo_model": COSMO_MODEL,
        "classy_sz_version": csz_version,
    }
    if save_to is not None:
        save_dataset(Path(save_to), params, cls,
                     lmax_cmb=lmax_cmb, lmax_pp=lmax_pp,
                     metadata={"classy_sz_version": csz_version,
                               "seed": seed, "cosmo_model": COSMO_MODEL})
    return out


def _predict_on_test(emulator: EmulatorProtocol, test: dict) -> dict[str, np.ndarray]:
    """Run emulator.predict() over every test-set parameter point.

    Returns a cls-dict with shapes matching test.
    """
    N = test["params"].shape[0]
    lmax_cmb = test["lmax_cmb"]
    lmax_pp = test["lmax_pp"]
    emu = {
        "tt": np.empty((N, lmax_cmb + 1), dtype=np.float32),
        "te": np.empty((N, lmax_cmb + 1), dtype=np.float32),
        "ee": np.empty((N, lmax_cmb + 1), dtype=np.float32),
        "pp": np.empty((N, lmax_pp + 1), dtype=np.float32),
    }
    for i in range(N):
        p = params_array_to_dict(test["params"][i])
        out = emulator.predict(p)
        emu["tt"][i] = out["tt"][: lmax_cmb + 1]
        emu["te"][i] = out["te"][: lmax_cmb + 1]
        emu["ee"][i] = out["ee"][: lmax_cmb + 1]
        emu["pp"][i] = out["pp"][: lmax_pp + 1]
    return emu


def get_accuracy_score(
    emulator: EmulatorProtocol,
    test: dict | None = None,
) -> dict[str, Any]:
    """Precision component of the competition score (no timing).

    Runs the emulator on every test parameter point, computes the Wishart
    chi^2 matrices (total + CMB block + PP block), and returns the MAE on
    Delta chi^2 for each. Fast and deterministic: call this on every
    training epoch.

    Parameters
    ----------
    emulator : object with a ``predict(params: dict) -> dict`` method.
    test : a test-set dict as returned by ``load_test``. If None, the
        default test set is downloaded via ``load_test``.

    Returns
    -------
    dict with keys:
        "mae_total" / "mae_cmb" / "mae_pp" : dicts with mae, median_abs_diff,
            max_abs_diff, n_pairs.
        "n_test", "lmax_cmb", "lmax_pp" : metadata.
    """
    if test is None:
        from .data import load_test
        test = load_test()

    emu_cls = _predict_on_test(emulator, test)

    split_true = chi2_matrix_split(test, test,
                                   lmax_cmb=test["lmax_cmb"],
                                   lmax_pp=test["lmax_pp"])
    split_emu = chi2_matrix_split(test, emu_cls,
                                  lmax_cmb=test["lmax_cmb"],
                                  lmax_pp=test["lmax_pp"])
    maes = chi2_mae_blocks(split_true, split_emu)

    return {
        "mae_total": maes["total"],
        "mae_cmb": maes["cmb"],
        "mae_pp": maes["pp"],
        "n_test": int(test["params"].shape[0]),
        "lmax_cmb": int(test["lmax_cmb"]),
        "lmax_pp": int(test["lmax_pp"]),
    }


def get_time_score(
    emulator: EmulatorProtocol,
    n_calls: int = 1000,
    n_warmup: int = 10,
    seed: int = 0,
    strict_cpu: bool = False,
) -> dict[str, Any]:
    """Speed component of the competition score (no precision work).

    Runs ``n_warmup`` untimed calls then ``n_calls`` timed sequential calls
    through ``emulator.predict`` with fresh LHC parameters. Returns the
    ``TimingResult`` contents as a plain dict.

    Pin JAX/TF to CPU before importing the emulator for a faithful MCMC
    benchmark:

        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=

    Pass ``strict_cpu=True`` to raise on non-CPU JAX backends instead of
    warning.

    Returns
    -------
    dict with keys:
        "t_cpu_ms_mean", "t_cpu_ms_median", "t_cpu_ms_std",
        "n_calls", "n_warmup", "jax_on_cpu".
    """
    timing = benchmark_emulator(emulator,
                                n_calls=n_calls,
                                n_warmup=n_warmup,
                                seed=seed,
                                strict_cpu=strict_cpu)
    return asdict(timing)


def get_score(
    emulator: EmulatorProtocol,
    test: dict | None = None,
    alpha: float = 1.0,
    t_floor_ms: float = 1.0,
    n_timing_calls: int = 1000,
    n_timing_warmup: int = 10,
    timing_seed: int = 0,
    measure_timing: bool = True,
    strict_cpu: bool = False,
) -> dict[str, Any]:
    """Run the full scorer on an emulator (precision + timing + combined).

    Thin wrapper around ``get_accuracy_score`` + ``get_time_score`` that also
    returns the combined score

        combined_S = log10(mae_total) + alpha * log10(max(t_cpu_ms, t_floor_ms)).

    Parameters
    ----------
    emulator : object with a ``predict(params: dict) -> dict`` method.
    test : a test-set dict as returned by ``load_test``. If None, the
        default test set is downloaded via ``load_test``.
    alpha : precision-vs-speed tradeoff in the combined score.
    t_floor_ms : soft floor (default 1 ms) on the timing term of
        ``combined_S``. Sub-millisecond inference does not buy further
        improvements in the MCMC workloads this benchmark targets.
    n_timing_calls, n_timing_warmup, timing_seed : benchmark knobs.
    measure_timing : set False to skip the timing harness entirely.
    strict_cpu : raise if JAX is not pinned to CPU.

    Returns
    -------
    dict with keys:
        "mae_total" / "mae_cmb" / "mae_pp"   : precision sub-scores
        "timing"                             : timing dict (or None)
        "combined_S"                         : float (or None)
        "alpha", "t_floor_ms"                : score hyper-parameters used
        "n_test", "lmax_cmb", "lmax_pp"      : dataset metadata
    """
    acc = get_accuracy_score(emulator, test=test)

    timing = None
    S = None
    if measure_timing:
        timing = get_time_score(emulator,
                                n_calls=n_timing_calls,
                                n_warmup=n_timing_warmup,
                                seed=timing_seed,
                                strict_cpu=strict_cpu)
        S = combined_score(acc["mae_total"]["mae"], timing["t_cpu_ms_mean"],
                           alpha=alpha, t_floor_ms=t_floor_ms)

    return {
        **acc,
        "timing": timing,
        "combined_S": S,
        "alpha": alpha,
        "t_floor_ms": t_floor_ms,
    }
