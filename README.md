# cmbemu

A CMB power-spectrum emulator competition: pre-generated training/test data,
Wishart-likelihood scoring, and a CPU timing harness built on
[`classy_sz`](https://pypi.org/project/classy_sz/) / `classy_szfast`.

## Install

```bash
pip install cmbemu
```

## Quickstart

```python
import cmbemu as cec

# (1) Fetch the pre-generated datasets (cached under ~/.cache/cmbemu)
train = cec.load_train()
test  = cec.load_test()

# (2) Optionally generate more spectra to study data-scaling
extra = cec.generate_data(n=20_000, seed=42, save_to="extra.npz")

# (3) Define your emulator
class MyEmulator:
    def predict(self, params: dict) -> dict:
        # params has keys: omega_b, omega_cdm, H0, tau_reio,
        # ln10^{10}A_s, n_s.
        # Returns a dict with keys tt, te, ee (len lmax_cmb+1)
        # and pp (len lmax_pp+1).
        ...

# (4) Score
result = cec.get_score(MyEmulator())
print(result["combined_S"], result["mae_total"]["mae"], result["timing"]["t_cpu_ms_mean"])
```

See `scoring.md` for the full likelihood and score definitions.

## Hardware note

Timing is done **CPU-only**, single-threaded warm sequential calls — this
mimics how an emulator is used inside a typical MCMC chain. Before running
`get_score` with `strict_cpu=True`, export:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
```

## Project layout

- `src/cmbemu/` — the installable package
- `scripts/` — internal dataset-building and diagnostic scripts
- `scoring.md` — formal score definition
- `STATUS.md` — current state of the competition design
