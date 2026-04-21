"""Benchmark classy_szfast spectrum generation: per-call time and per-spectrum memory."""
import gc
import time

import numpy as np
from classy_sz import Class as Class_sz

PLANCK = {
    "omega_b": 0.02242,
    "omega_cdm": 0.11933,
    "H0": 67.66,
    "tau_reio": 0.0561,
    "ln10^{10}A_s": 3.047,
    "n_s": 0.9665,
}

csz = Class_sz()
t0 = time.perf_counter()
csz.initialize_classy_szfast()
t_init = time.perf_counter() - t0

# Warm up
out = csz.get_cmb_cls(params_values_dict=PLANCK)

ell = out["ell"]
print(f"init: {t_init:.2f}s")
print(f"returned keys: {sorted(out.keys())}")
print(f"ell shape {ell.shape}, ell min {ell.min()}, ell max {ell.max()}")
for k in ("tt", "te", "ee", "pp"):
    a = out[k]
    print(f"  {k}: shape {a.shape}, dtype {a.dtype}, nbytes {a.nbytes}")

bytes_per_spec_full = sum(out[k].nbytes for k in ("tt", "te", "ee", "pp"))
bytes_per_spec_f32 = bytes_per_spec_full // 2
print(f"per-spectrum bytes (tt+te+ee+pp, float64): {bytes_per_spec_full}")
print(f"  projected 10k spectra: {bytes_per_spec_full*10_000/1e9:.3f} GB (float64), "
      f"{bytes_per_spec_f32*10_000/1e9:.3f} GB (float32)")
print(f"  projected 50k spectra: {bytes_per_spec_full*50_000/1e9:.3f} GB (float64), "
      f"{bytes_per_spec_f32*50_000/1e9:.3f} GB (float32)")

N = 50
rng = np.random.default_rng(0)
# jitter params by ~1% to avoid any trivial caching
keys = list(PLANCK.keys())
params_list = []
for _ in range(N):
    p = dict(PLANCK)
    for k in keys:
        p[k] = PLANCK[k] * (1.0 + 0.01 * rng.standard_normal())
    params_list.append(p)

gc.collect()
t0 = time.perf_counter()
for p in params_list:
    _ = csz.get_cmb_cls(params_values_dict=p)
dt = time.perf_counter() - t0
per_call = dt / N
print(f"\n{N} calls: total {dt:.2f}s, per-call {per_call*1000:.1f} ms")
print(f"  projected 10k: {per_call*10_000/60:.1f} min")
print(f"  projected 50k: {per_call*50_000/60:.1f} min")
