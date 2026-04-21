# Data description

This document describes the dataset distributed with the `cmbemu` competition: what is in it, how to load it, how to generate more of it if you want to study data-scaling, and how to score an emulator against it.

Everything here is reproducible from the `cmbemu` Python package (`pip install cmbemu`). The raw files live on the Hugging Face Hub at
[`borisbolliet/cmbemu-competition-v1`](https://huggingface.co/datasets/borisbolliet/cmbemu-competition-v1) and are cached locally under `~/.cache/cmbemu/` on first use.

## Files

| File              | N      | Size (float32)  | Purpose                              |
|-------------------|:------:|:---------------:|--------------------------------------|
| `train.npz`       | 50 000 | ~3.95 GB        | Training set for emulator authors    |
| `test.npz`        |  5 000 | ~0.38 GB        | Held-out scoring test set            |
| `train_small.npz` |  5 000 | ~0.38 GB        | Dev-loop training set                |
| `test_small.npz`  |    500 | ~38 MB          | Dev-loop test set                    |

All four files share the same schema. The `*_small` variants exist purely so you can iterate on your training script without pulling 4 GB.

## Schema

Each `.npz` contains the following arrays:

| Key                 | Shape              | dtype    | Description                                                        |
|---------------------|--------------------|:--------:|--------------------------------------------------------------------|
| `params`            | (N, 6)             | float32  | LHC points in physical units                                       |
| `param_names`       | (6,)               | str      | Ordering of the parameter columns                                  |
| `tt`                | (N, 6001)          | float32  | $C_\ell^{TT}$ for $\ell \in [0, 6000]$ (same convention as `te`, `ee`) |
| `te`                | (N, 6001)          | float32  | $C_\ell^{TE}$                                                      |
| `ee`                | (N, 6001)          | float32  | $C_\ell^{EE}$                                                      |
| `pp`                | (N, 3001)          | float32  | $C_\ell^{\phi\phi}$ for $\ell \in [0, 3000]$                      |
| `box_lo`, `box_hi`  | (6,)               | float32  | Competition box bounds                                             |
| `lmax_cmb`, `lmax_pp` | ()               | int32    | 6000, 3000                                                         |
| `cosmo_model`       | ()                 | int32    | 6 (= `ede-v2` in `classy_szfast`)                                  |
| `classy_sz_version` | ()                 | str      | Exact generator version                                            |
| `seed`              | ()                 | int64    | RNG seed used for the LHC                                          |

### Conventions

- `params` columns are ordered as `("omega_b", "omega_cdm", "H0", "tau_reio", "ln10^{10}A_s", "n_s")`. This ordering is authoritative; use `cmbemu.PARAM_NAMES` rather than hard-coding it.
- All $\ell$-indexed arrays start at $\ell = 0$; the monopole and dipole are kept for convenience but are ignored by the likelihood — slice from $\ell = 2$ when scoring.
- Spectra are in whatever units `classy_szfast` returns (SI, $C_\ell$ as is, not $D_\ell$). **This does not matter for the likelihood**, which is unit-invariant; see [`scoring.md`](scoring.md).

## Loading

### High-level API

```python
import cmbemu as cec

train = cec.load_train()   # downloads + caches on first call
test  = cec.load_test()

N = train["params"].shape[0]     # 50000
lmax_cmb = train["lmax_cmb"]     # 6000
tt = train["tt"]                 # (N, lmax_cmb+1) float32
params = train["params"]         # (N, 6)  float32
```

For the dev subsets, pass `size="small"`:

```python
train_small = cec.load_train(size="small")
test_small  = cec.load_test(size="small")
```

### Direct `numpy.load`

If you want to inspect the file without the `cmbemu` package:

```python
import numpy as np
with np.load("train.npz", allow_pickle=False) as z:
    params = z["params"]
    tt, te, ee = z["tt"], z["te"], z["ee"]
    pp = z["pp"]
    param_names = [str(s) for s in z["param_names"]]
    cosmo_model = int(z["cosmo_model"])
```

## Parameter box

The Latin-hypercube covers Planck 2018 to ≥10σ on every axis, well inside the `classy_szfast` ede-v2 emulator validity region:

| Parameter    | Min    | Max    |
|--------------|:------:|:------:|
| ω_b          | 0.020  | 0.025  |
| ω_cdm        | 0.09   | 0.15   |
| H₀           | 55     | 85     |
| τ            | 0.03   | 0.10   |
| ln 10¹⁰ A_s  | 2.7    | 3.3    |
| n_s          | 0.92   | 1.02   |

Each LHC point is a single cosmology; the corresponding spectrum rows in `tt`, `te`, `ee`, `pp` share an index.

## Generating more data

Participants are encouraged to study how emulator precision scales with $N_\text{train}$. The `cmbemu.generate_data` helper produces fresh spectra from the same LHC prior:

```python
extra = cec.generate_data(
    n=20_000,
    seed=42,                    # any seed; pick something not in {202604, 202605}
    save_to="extra_20k.npz",
)
```

The output dict has the same schema as `train.npz` / `test.npz`. Generation is single-process, ~5 ms per spectrum on recent Apple Silicon / Ryzen CPUs — 20 k spectra is ~2 minutes. The script is not resumable; for very large runs we recommend splitting into multiple invocations with distinct seeds and concatenating arrays afterwards.

**Never** regenerate using a seed in `{202604, 202605}` — those are the train/test seeds and would reproduce the held-out test set.

## Computing the likelihood

The competition uses the CV-limited, full-sky Wishart log-likelihood — see [`scoring.md`](scoring.md) for the formula and its derivation. In code:

```python
from cmbemu.likelihood import chi2_matrix_split, chi2_matrix, chi2_diag

# All NxN pairs, split into CMB (TT/TE/EE 2x2 Wishart) and PP (1x1)
split = chi2_matrix_split(data_cls=test, theory_cls=test)
chi2_cmb   = split["cmb"]      # (N, N)
chi2_pp    = split["pp"]       # (N, N)
chi2_total = split["total"]    # cmb + pp

# Shortcut that returns only the total
chi2 = chi2_matrix(data_cls=test, theory_cls=test)

# Diagonal only (self-chi2 when theory == data): useful as a cheap diagnostic.
# By construction chi2_diag(d, d) ~ 0 (float64 roundoff only).
diag = chi2_diag(test, test)
```

Arguments default to `lmax_cmb=6000`, `lmax_pp=3000`, `f_sky=1.0` — matching the dataset.

## Scoring an emulator end-to-end

The full scorer:

```python
class MyEmulator:
    def predict(self, params: dict) -> dict:
        return {"tt": ..., "te": ..., "ee": ..., "pp": ...}

result = cec.get_score(MyEmulator())
# Returns:
#   {"mae_total": {"mae": ..., "median_abs_diff": ..., "max_abs_diff": ..., "n_pairs": ...},
#    "mae_cmb":   {...},   # TT/TE/EE 2x2 Wishart block only
#    "mae_pp":    {...},   # phi-phi only
#    "timing":    {"t_cpu_ms_mean": ..., "t_cpu_ms_median": ..., ...} | None,
#    "combined_S": ... | None,
#    "alpha": 1.0, "n_test": 5000, ...}
```

The precision block runs in O(seconds) — we vectorize the likelihood across `(i, j, ℓ)` via BLAS matmul — so scoring a new emulator is dominated by the time it takes to call `emulator.predict` 5 000 times, plus the timing harness (1 000 warm sequential calls by default). On a typical laptop the whole call completes in under a minute.

### Measuring speed honestly

Timing is done CPU-only, single-threaded, warm, with fresh LHC parameters at every call. Set these before importing your emulator:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
```

Pass `strict_cpu=True` to `cec.get_score` to make this a hard requirement rather than a warning.

## Minimal working example

```python
import cmbemu as cec

# 1) get the test set (downloaded on first call)
test = cec.load_test()

# 2) a trivial reference emulator
emu = cec.ConstantPlanck(lmax_cmb=test["lmax_cmb"], lmax_pp=test["lmax_pp"])

# 3) score it
result = cec.get_score(emu, test=test, n_timing_calls=200)

print(f"mae_total  : {result['mae_total']['mae']:.3e}")
print(f"mae_cmb    : {result['mae_cmb']['mae']:.3e}")
print(f"mae_pp     : {result['mae_pp']['mae']:.3e}")
print(f"t_cpu_ms   : {result['timing']['t_cpu_ms_mean']:.4g}")
print(f"combined_S : {result['combined_S']:.3f}")
```

Expected numbers on the full 5 000-point test set: `mae_total ≈ 1.13e7`, `t_cpu_ms ≈ 1.5e-3` (ConstantPlanck is essentially a dict copy), `combined_S ≈ 4.23`. Any real emulator should beat this floor on precision.
