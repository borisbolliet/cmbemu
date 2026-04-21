# Data description

Everything you need to know about the competition dataset: what is on disk, what each array means, and how to load it. Assumes you have already run `pip install cmbemu` and that the data is available through `cmbemu.load_train()` / `cmbemu.load_test()`.

## Files

The competition ships four `.npz` files. The full dataset is the one to use for real training and scoring; the `*_small` variants exist so you can iterate on a training script without loading 4 GB into memory.

| File              | N      | Size (float32) | Purpose                             |
|-------------------|:------:|:--------------:|-------------------------------------|
| `train.npz`       | 50 000 | ~4 GB          | Training set                        |
| `test.npz`        |  5 000 | ~0.4 GB        | Held-out scoring test set           |
| `train_small.npz` |  5 000 | ~0.4 GB        | Dev-loop training set               |
| `test_small.npz`  |    500 | ~40 MB         | Dev-loop test set                   |

All four share an identical schema.

## Loading

### The one-liners

```python
import cmbemu as cec

train = cec.load_train()          # full 50 000-point training set
test  = cec.load_test()           # full 5 000-point test set

train_small = cec.load_train(size="small")
test_small  = cec.load_test(size="small")
```

Each call returns a plain dict of typed numpy arrays.

### Direct `numpy.load`

You don't need `cmbemu` to read the files:

```python
import numpy as np

with np.load("test.npz", allow_pickle=False) as z:
    params = z["params"]          # (N, 6)
    tt, te, ee = z["tt"], z["te"], z["ee"]
    pp = z["pp"]
    param_names = tuple(str(s) for s in z["param_names"])
    lmax_cmb = int(z["lmax_cmb"])
    lmax_pp  = int(z["lmax_pp"])
```

## Schema

Every file contains:

| Key                 | Shape              | dtype    | Description                                                     |
|---------------------|--------------------|:--------:|-----------------------------------------------------------------|
| `params`            | (N, 6)             | float32  | LHC points in physical units                                    |
| `param_names`       | (6,)               | str      | Canonical ordering of the parameter columns                     |
| `tt`                | (N, 6001)          | float32  | $C_\ell^{TT}$ for $\ell \in [0, 6000]$                          |
| `te`                | (N, 6001)          | float32  | $C_\ell^{TE}$                                                   |
| `ee`                | (N, 6001)          | float32  | $C_\ell^{EE}$                                                   |
| `pp`                | (N, 3001)          | float32  | $C_\ell^{\phi\phi}$ for $\ell \in [0, 3000]$                    |
| `box_lo`, `box_hi`  | (6,)               | float32  | Parameter box bounds (matches the Parameter prior table below)  |
| `lmax_cmb`          | ()                 | int32    | 6000                                                            |
| `lmax_pp`           | ()                 | int32    | 3000                                                            |
| `seed`              | ()                 | int64    | RNG seed used for the LHC                                       |

Key `i` of every array corresponds to the same cosmology: `params[i]` is the parameter vector, and `tt[i]`, `te[i]`, `ee[i]`, `pp[i]` are its four spectra.

## Parameter columns

`params` rows are ordered exactly as in `param_names`. Use `cmbemu.PARAM_NAMES` (which is the same tuple) rather than hard-coding indices:

```python
>>> cmbemu.PARAM_NAMES
('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
```

Conversion between a parameter row and the `dict` your emulator sees:

```python
from cmbemu import PARAM_NAMES, params_array_to_dict

p_dict = params_array_to_dict(params[i])
# {"omega_b": ..., "omega_cdm": ..., "H0": ..., "tau_reio": ...,
#  "ln10^{10}A_s": ..., "n_s": ...}
```

## Parameter prior

Latin-hypercube sampled. Every training and testing cosmology lies strictly inside this 6-D box:

| Parameter     | Min    | Max    |
|---------------|:------:|:------:|
| ω_b           | 0.020  | 0.025  |
| ω_cdm         | 0.09   | 0.15   |
| H₀            | 55     | 85     |
| τ             | 0.03   | 0.10   |
| ln 10¹⁰ A_s   | 2.7    | 3.3    |
| n_s           | 0.92   | 1.02   |

Your emulator only needs to be accurate inside this box. Nothing in the scorer will query it outside these bounds.

## Multipole conventions

All spectrum arrays are indexed by $\ell$, **starting at $\ell = 0$**:

- `tt[i, ell]`, `te[i, ell]`, `ee[i, ell]` are defined for $\ell \in [0, 6000]$ (length 6001).
- `pp[i, ell]` is defined for $\ell \in [0, 3000]$ (length 3001).

The $\ell = 0$ and $\ell = 1$ entries are included for convenience but are ignored everywhere in the scoring — the Wishart likelihood starts at $\ell = 2$. Keep them in your emulator's output; the scorer slices them off.

$C_\ell$ units are whatever convention was used to generate the data (SI, $C_\ell$ as is, not $D_\ell$). **This does not affect the score**: the Wishart likelihood is invariant under any consistent rescaling of $C_\ell$ (the 2×2 Wishart `Tr(C_d C_t^{-1})` and `log det` both cancel scale factors), so your emulator may emit $C_\ell$, $D_\ell = \ell(\ell+1)C_\ell/(2\pi)$, or anything else, as long as it applies the same convention for every sample and every multipole.

## Size reference

Just so you know what fits where:

- `train.npz`: 4.0 GB on disk, 6.3 GB in memory (all four spectra, float32).
  - `params`: 1.2 MB
  - `tt`/`te`/`ee`: 1.2 GB each (50 000 × 6001 × 4 bytes)
  - `pp`: 0.6 GB
- `test.npz`: 0.4 GB on disk, 0.6 GB in memory.
- The $5000 \times 5000$ scoring chi² matrix allocated during `get_score` is another 0.2 GB (float64). Peak RAM while scoring is ~2 GB — fits comfortably in a 16 GB laptop.

## Generating more data (optional)

The training and test sets released here are fixed; you should not extend the test set. But if you want to study how your architecture's precision scales with $N_\text{train}$, you can draw additional cosmologies from the same prior:

```python
extra = cec.generate_data(
    n=20_000,
    seed=42,                    # any integer not in {202604, 202605}
    save_to="extra_20k.npz",
)
```

The returned dict has the same schema, and the saved `.npz` can be loaded with `numpy.load` or merged into your training pipeline. Seeds `202604` and `202605` are reserved for the train and test sets; using either of them would reproduce the released data exactly, which is not what you want.

## Sanity check

Once your install is working, this 6-line script confirms the dataset is reachable and the scoring interface is live:

```python
import cmbemu as cec
test = cec.load_test()
emu  = cec.ConstantPlanck(lmax_cmb=test["lmax_cmb"], lmax_pp=test["lmax_pp"])
result = cec.get_score(emu, test=test)
print(result["combined_S"], result["mae_total"]["mae"])
# Expect combined_S ~ 7.05, mae_total ~ 1.13e7
```

If those numbers come out close to the expected values, you're ready to swap in a real emulator.
