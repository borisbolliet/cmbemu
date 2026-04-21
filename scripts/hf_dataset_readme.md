---
license: mit
task_categories:
  - other
tags:
  - cosmology
  - CMB
  - emulator
  - benchmark
  - power-spectrum
size_categories:
  - 10K<n<100K
---

# cmbemu competition v1 — CMB power-spectrum emulator benchmark

Pre-generated CMB angular power spectra for a Latin-hypercube sweep over a 6-parameter cosmology box, produced by [`classy_szfast`](https://github.com/CLASS-SZ/classy_szfast) with `cosmo_model=6` (ede-v2 emulator stack, near-LCDM in this configuration). Intended as the official data distribution for the `cmbemu` emulator-competition package.

## Contents

| File              | N      | Size (float32) | Purpose |
|-------------------|:------:|:--------------:|---------|
| `train.npz`       | 50 000 | ~3.7 GB        | Training set for emulator authors |
| `test.npz`        |  5 000 | ~0.4 GB        | Scoring test set |
| `train_small.npz` |  5 000 | ~0.4 GB        | Dev-loop training set |
| `test_small.npz`  |    500 | ~40 MB         | Dev-loop test set |

All four files share the same schema.

## Schema

Each `.npz` contains:

| Key                 | Shape                  | dtype    | Description |
|---------------------|------------------------|:--------:|-------------|
| `params`            | (N, 6)                 | float32  | LHC points in physical units |
| `param_names`       | (6,)                   | str      | Order of the 6 cosmology parameters |
| `tt`                | (N, 6001)              | float32  | C_ℓ^TT for ℓ ∈ [0, 6000] |
| `te`                | (N, 6001)              | float32  | C_ℓ^TE |
| `ee`                | (N, 6001)              | float32  | C_ℓ^EE |
| `pp`                | (N, 3001)              | float32  | C_ℓ^{φφ} for ℓ ∈ [0, 3000] |
| `box_lo`, `box_hi`  | (6,)                   | float32  | Competition box bounds |
| `lmax_cmb`, `lmax_pp` | ()                   | int32    | 6000, 3000 |
| `cosmo_model`       | ()                     | int32    | 6 (= `ede-v2` in classy_szfast) |
| `classy_sz_version` | ()                     | str      | Exact generator version |
| `seed`              | ()                     | int64    | RNG seed used for the LHC |

All ℓ-indexed arrays start at ℓ=0. The monopole/dipole (ℓ=0, 1) are ignored by the likelihood — slice from ℓ=2 when scoring.

## Parameter box

The 6D competition prior covers roughly ±10σ–30σ around Planck 2018, strictly inside the `classy_szfast` ede-v2 emulator validity region:

| Parameter       | Min    | Max    |
|-----------------|:------:|:------:|
| ω_b             | 0.020  | 0.025  |
| ω_cdm           | 0.09   | 0.15   |
| H₀              | 55     | 85     |
| τ               | 0.03   | 0.10   |
| ln 10¹⁰ A_s     | 2.7    | 3.3    |
| n_s             | 0.92   | 1.02   |

## Quickstart

```python
import cmbemu as cec

# Downloads the .npz files on first use, cached under ~/.cache/cmbemu
train = cec.load_train()
test  = cec.load_test()

# Score your emulator
class MyEmulator:
    def predict(self, params: dict) -> dict:
        ...

result = cec.get_score(MyEmulator())
print(result["combined_S"], result["mae_total"]["mae"])
```

See the [cmbemu repository](https://github.com/borisbolliet/cmbemu) for the full scoring definition and competition rules.

## Generation reproducibility

Regenerate any subset locally:

```python
import cmbemu as cec
data = cec.generate_data(n=5000, seed=202605, save_to="test_reproduced.npz")
```

Seeds used here: `202604` for train, `202605` for test.

## License

MIT.
