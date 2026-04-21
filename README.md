# cmbemu

A CMB power-spectrum emulator competition: pre-generated training/test data,
Wishart-likelihood scoring, and a CPU timing harness built on
[`classy_sz`](https://pypi.org/project/classy_sz/) / `classy_szfast`.

## Install

```bash
pip install cmbemu
```

Requires Python ≥ 3.12.

## Quickstart

```python
import cmbemu as cec

# 1. Fetch the competition data (cached under ~/.cache/cmbemu)
train = cec.load_train()
test  = cec.load_test()

# 2. Define your emulator: any object exposing predict(dict) -> dict
class MyEmulator:
    def predict(self, params: dict):
        # params keys: omega_b, omega_cdm, H0, tau_reio, ln10^{10}A_s, n_s
        # return dict with keys tt/te/ee (len 6001) and pp (len 3001)
        ...

emu = MyEmulator()

# 3. Score it
result = cec.get_score(emu)
print(result["combined_S"], result["mae_total"]["mae"])
```

See [`idea.md`](idea.md) for the competition rules and
[`data_description.md`](data_description.md) for the dataset schema.

## The scoring API

Three parallel entry points, one per component of the score:

| Function                | Returns                               | Cost per call          | When to use |
|-------------------------|---------------------------------------|------------------------|-------------|
| `get_accuracy_score`    | `mae_total`, `mae_cmb`, `mae_pp`      | ~2 s (run emulator on the 5 000 test points) | every training epoch |
| `get_time_score`        | `t_cpu_ms_mean/median/std`, ...       | ~`n_calls × per-call ms` | after the model is frozen |
| `get_score`             | both of the above **plus** `combined_S` | sum of the two         | at submission time |

```python
acc  = cec.get_accuracy_score(emu)       # precision only
tim  = cec.get_time_score(emu)           # timing only
full = cec.get_score(emu)                # both + combined_S
```

### What `combined_S` is

The single scalar leaderboard number. With $E$ = `mae_total` (the precision MAE),
$T$ = mean CPU time per call in ms, $T_0$ = `t_floor_ms` (default 1 ms),
$\alpha$ = `alpha` (default 1):

$$
S \;=\; \log_{10}(E) \,+\, \alpha \cdot \log_{10}(\max(T,\, T_0))
$$

Lower is better. Above the floor, one decade of precision trades for one
decade of speed; sub-millisecond inference buys no further credit (MCMC
per-step overhead dominates below that scale).

Precision and speed are cleanly decomposed: you can recompute `combined_S`
from the two sub-scores yourself:

```python
acc = cec.get_accuracy_score(emu)
tim = cec.get_time_score(emu)
S   = cec.combined_score(acc["mae_total"]["mae"], tim["t_cpu_ms_mean"])
```

See [`idea.md`](idea.md#scoring) for the full Wishart derivation, block
decomposition, and rationale for MAE over MSE.

## CPU timing setup

Timing is done CPU-only, single-threaded, warm, with fresh parameters at
every call (no result caching possible). Before importing your emulator:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
```

Pass `strict_cpu=True` to `get_time_score` / `get_score` to make the check a
hard failure rather than a warning.

## Generate more data (optional)

Want to study how your architecture's precision scales with $N_\text{train}$?
Draw more cosmologies from the same prior:

```python
extra = cec.generate_data(n=20_000, seed=42, save_to="extra.npz")
```

## Baselines

Two trivial emulators ship with the package for sanity checks:

```python
emu1 = cec.ConstantPlanck()                    # always returns Planck fiducial
emu2 = cec.NearestNeighbour(cec.load_train())  # 1-NN lookup on training set
```

Expected `ConstantPlanck` → `combined_S ≈ 7.05` on the full test set. Any
real emulator should beat this on precision by many decades.

## Project layout

- `src/cmbemu/` — the installable package
- `scripts/` — internal dataset-building and diagnostic tools
- `idea.md` — competition pitch
- `data_description.md` — dataset schema + loading patterns
- `STATUS.md` — current state of the competition design
