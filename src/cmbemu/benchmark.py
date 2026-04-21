"""CPU-pinned timing harness for the speed component of the score.

The competition scoring container provides 1 GPU and 64 CPUs, but MCMC chains
are single-threaded per chain. Timing is therefore done with the submission's
`predict` called sequentially and warm. The container is responsible for
pinning JAX/TF to CPU *before* the process starts (set
``JAX_PLATFORMS=cpu`` and ``CUDA_VISIBLE_DEVICES=``), since those env vars
must be set before the ML framework is imported.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import numpy as np

from .box import PARAM_NAMES, box_bounds, params_array_to_dict


@dataclass(frozen=True)
class TimingResult:
    t_cpu_ms_mean: float
    t_cpu_ms_median: float
    t_cpu_ms_std: float
    n_calls: int
    n_warmup: int
    jax_on_cpu: bool


def _jax_on_cpu() -> bool:
    """Best-effort check that JAX is running CPU-only."""
    if os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
        return True
    try:
        import jax  # type: ignore
        devs = jax.devices()
        return all(d.platform == "cpu" for d in devs)
    except Exception:
        return True  # no JAX imported — treat as CPU


def benchmark_emulator(
    emulator,
    n_calls: int = 1000,
    n_warmup: int = 10,
    seed: int = 0,
    strict_cpu: bool = False,
) -> TimingResult:
    """Measure mean wall time (ms) per ``predict`` call.

    - Samples n_calls + n_warmup fresh LHC points in the competition box.
    - Runs n_warmup calls to absorb JIT compile / lazy init.
    - Times each of the remaining n_calls calls separately so we can report
      the mean, median, and standard deviation.

    Parameters
    ----------
    emulator : object with a ``predict(params: dict) -> dict`` method
    n_calls : int
    n_warmup : int
    seed : int
    strict_cpu : if True, raise when JAX is not pinned to CPU.
    """
    cpu_ok = _jax_on_cpu()
    if strict_cpu and not cpu_ok:
        raise RuntimeError(
            "JAX is not pinned to CPU. Set JAX_PLATFORMS=cpu and "
            "CUDA_VISIBLE_DEVICES= before importing your emulator."
        )
    if not cpu_ok:
        warnings.warn(
            "JAX appears to be using a non-CPU backend. For MCMC-realistic "
            "timing, set JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES= before "
            "importing the emulator.", RuntimeWarning, stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    lo, hi = box_bounds()
    u = rng.random((n_calls + n_warmup, len(PARAM_NAMES)))
    pts = lo + u * (hi - lo)

    for i in range(n_warmup):
        emulator.predict(params_array_to_dict(pts[i]))

    times_ns = np.empty(n_calls, dtype=np.int64)
    for i in range(n_calls):
        p = params_array_to_dict(pts[n_warmup + i])
        t0 = time.perf_counter_ns()
        emulator.predict(p)
        times_ns[i] = time.perf_counter_ns() - t0

    t_ms = times_ns * 1e-6
    return TimingResult(
        t_cpu_ms_mean=float(t_ms.mean()),
        t_cpu_ms_median=float(np.median(t_ms)),
        t_cpu_ms_std=float(t_ms.std()),
        n_calls=n_calls,
        n_warmup=n_warmup,
        jax_on_cpu=cpu_ok,
    )
