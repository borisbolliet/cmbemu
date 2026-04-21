"""CPU-only throughput for classy_szfast: sequential warm calls, varying params.

Emulates the inner loop of a Metropolis MCMC on CPU.
"""
import os
# Force JAX to CPU so we measure a realistic MCMC runtime environment
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import numpy as np

from classy_sz import Class as Class_sz

PLANCK = {
    "omega_b": 0.02242, "omega_cdm": 0.11933, "H0": 67.66,
    "tau_reio": 0.0561, "ln10^{10}A_s": 3.047, "n_s": 0.9665,
}

csz = Class_sz()
csz.initialize_classy_szfast()

# Warm up
_ = csz.get_cmb_cls(params_values_dict=PLANCK)
_ = csz.get_cmb_cls(params_values_dict=PLANCK)

rng = np.random.default_rng(0)
keys = list(PLANCK.keys())

for N in (100, 1000):
    params = []
    for _ in range(N):
        p = dict(PLANCK)
        for k in keys:
            p[k] = PLANCK[k] * (1.0 + 0.01 * rng.standard_normal())
        params.append(p)
    t0 = time.perf_counter()
    for p in params:
        _ = csz.get_cmb_cls(params_values_dict=p)
    dt = time.perf_counter() - t0
    per = dt / N
    print(f"{N:>4} calls: total {dt:.3f}s, per-call {per*1e3:.2f} ms  ({1/per:.0f} Hz)")
