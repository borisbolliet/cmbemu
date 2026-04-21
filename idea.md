# cmbemu — Emulator Competition

## The task

Train a neural-network emulator of the four CMB angular power spectra — **TT**, **TE**, **EE**, and lensing-potential **φφ** — as a function of six ΛCDM cosmological parameters. Optimize for **both** precision (how close your spectra are to the reference at the likelihood level) **and** CPU inference speed (sub-ms `predict()` calls are the realistic target for use inside an MCMC chain).

You will be scored by a single function call — `cmbemu.get_score(emulator)` — which returns a precision number, a CPU timing number, and a combined score. All three are computed on a held-out 5 000-point test set that ships with the competition data.

## Inputs and outputs

### Input to your emulator

A Python dict with six keys, in physical units:

```python
{
    "omega_b":       float,   # Ω_b h²
    "omega_cdm":     float,   # Ω_cdm h²
    "H0":            float,   # km/s/Mpc
    "tau_reio":      float,
    "ln10^{10}A_s":  float,
    "n_s":           float,
}
```

Use `cmbemu.PARAM_NAMES` as the canonical ordering (same order as the rows above).

### Output from your emulator

A Python dict with four keys, each a `numpy.ndarray`:

```python
{
    "tt": array of shape (6001,),   # ell = 0, 1, ..., 6000
    "te": array of shape (6001,),
    "ee": array of shape (6001,),
    "pp": array of shape (3001,),   # ell = 0, 1, ..., 3000
}
```

Shapes and ordering are fixed. `ell=0, 1` entries must be present but are ignored by the scorer — slice them to zero if convenient. Units may be whatever you like; the scoring is unit-invariant in any consistent rescaling of $C_\ell$, so you can emulate $D_\ell = \ell(\ell+1)C_\ell/(2\pi)$ and rescale, or emulate $C_\ell$ directly — either is fine as long as you do the same thing for every multipole of every spectrum.

## The interface you must implement

Any object with a `predict(params: dict) -> dict` method. Example:

```python
import numpy as np

class MyEmulator:
    def __init__(self, weights_path):
        self.params_order = ("omega_b", "omega_cdm", "H0",
                             "tau_reio", "ln10^{10}A_s", "n_s")
        self.model = load_your_model(weights_path)

    def predict(self, params: dict) -> dict[str, np.ndarray]:
        x = np.array([params[k] for k in self.params_order])
        # ... your neural net call here ...
        return {"tt": tt, "te": te, "ee": ee, "pp": pp}
```

That is all the scorer requires — no base class, no registration, no extra methods.

## Scoring

Full definition in [`scoring.md`](scoring.md). Short version:

For every ordered pair $(i, j)$ of test-set cosmologies, the scorer treats $C_\ell^{\text{true}}(\theta_i)$ as mock data and evaluates the full-sky, cosmic-variance-limited Wishart log-likelihood with theory = `true(θ_j)` and with theory = `emulator(θ_j)`. The difference between those two χ²s measures the emulator's likelihood-level error.

- **Primary precision score** (lower is better):
  $$S_{\text{prec}} = \frac{1}{N(N-1)} \sum_{i \neq j} \bigl|\, \chi^2_{\text{emu}}(i,j) - \chi^2_{\text{true}}(i,j) \,\bigr|$$
  i.e. **MAE on Δχ²** over all 24 995 000 off-diagonal pairs. No cut — uses every pair.
- **Speed score**:
  $T_{\text{CPU, ms}}$ = mean wall time per `predict()` call, averaged over 1 000 warm sequential calls with fresh LHC parameters, with JAX/TF pinned to CPU.
- **Combined score** (lower is better):
  $$S = \log_{10}\!\bigl(S_{\text{prec}}\bigr) + \alpha\,\log_{10}\!\bigl(\max(T_{\text{CPU, ms}},\; T_{\text{floor}})\bigr),\qquad \alpha = 1,\quad T_{\text{floor}} = 1\ \text{ms}.$$
  Above the floor, halving precision error compensates doubling inference
  time, and vice versa. **Sub-millisecond inference buys no further speed
  credit** — MCMC per-step overhead dominates below ~1 ms, so additional
  nanoseconds have no effect on chain throughput. Same-precision emulators
  still break ties on speed, but only down to the floor.

## How to run the scorer

The scoring is exposed as **three parallel entry points**, one per piece of the competition score. Use whichever you need:

```python
import cmbemu as cec
emu = MyEmulator(weights_path="...")

acc  = cec.get_accuracy_score(emu)   # precision only  (~2 s)
tim  = cec.get_time_score(emu)       # timing only     (~n_calls × per-call ms)
full = cec.get_score(emu)            # both + combined_S
```

All three take the same `emulator` object. Under the hood, `get_score` just calls the other two and adds the combined scalar — no independent work.

### `get_accuracy_score(emu, test=None)`

Returns the **precision half** of the score. Runs `emu.predict` on all 5 000 test cosmologies, computes the Wishart $\chi^2$ matrices, and reports MAE per block. No timing harness.

```python
acc = cec.get_accuracy_score(emu)
# {
#   "mae_total": {"mae": ..., "median_abs_diff": ..., "max_abs_diff": ..., "n_pairs": ...},
#   "mae_cmb":   {...},   # TT/TE/EE 2x2 Wishart block only
#   "mae_pp":    {...},   # phi-phi only
#   "n_test": 5000, "lmax_cmb": 6000, "lmax_pp": 3000,
# }
```

**Use this during training** — it's deterministic, has no stochastic timing overhead, and gives you the single knob you can actually optimize against.

### `get_time_score(emu, n_calls=1000, ...)`

Returns the **speed half** of the score. Runs `n_warmup=10` untimed warmup calls, then `n_calls=1000` timed sequential calls with fresh LHC parameters, and reports mean/median/std wall time.

```python
tim = cec.get_time_score(emu, n_calls=1000, strict_cpu=True)
# {
#   "t_cpu_ms_mean":   ...,
#   "t_cpu_ms_median": ...,
#   "t_cpu_ms_std":    ...,
#   "n_calls":  1000,
#   "n_warmup": 10,
#   "jax_on_cpu": True,
# }
```

**Use this after freezing your model** to verify the inference-speed budget.

### `get_score(emu, alpha=1.0, t_floor_ms=1.0, ...)`

Computes both sub-scores above, then the **combined scalar** leaderboard number:

$$
S = \log_{10}(\text{mae\_total}) + \alpha \log_{10}\!\bigl(\max(t_{\text{cpu\_ms, mean}},\; T_{\text{floor}})\bigr)
$$

```python
full = cec.get_score(emu)
# {
#   "mae_total": {...}, "mae_cmb": {...}, "mae_pp": {...},     # from get_accuracy_score
#   "timing":    {...},                                         # from get_time_score
#   "combined_S": ...,                                          # the scalar leaderboard entry
#   "alpha": 1.0, "t_floor_ms": 1.0,
#   "n_test": 5000, "lmax_cmb": 6000, "lmax_pp": 3000,
# }
```

You can verify this by hand:

```python
acc = cec.get_accuracy_score(emu)
tim = cec.get_time_score(emu)
S   = cec.combined_score(acc["mae_total"]["mae"], tim["t_cpu_ms_mean"])
# S == cec.get_score(emu)["combined_S"]  (up to timing noise across reruns)
```

### Typical training loop

```python
for epoch in range(N_EPOCHS):
    train_step(emu)
    if epoch % 5 == 0:
        acc = cec.get_accuracy_score(emu)
        log(epoch, acc["mae_total"]["mae"])

# once satisfied, measure speed and get the submission number
final = cec.get_score(emu)
print(f"submission combined_S = {final['combined_S']:.3f}")
```

### CPU pinning for speed timing

Timing is done single-threaded on CPU. **Before importing your emulator or JAX**, set:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
```

Pass `strict_cpu=True` to `get_score` to make this a hard requirement rather than a warning:

```python
result = cec.get_score(emu, strict_cpu=True)
```

Skip timing entirely (useful while iterating on precision) with `measure_timing=False`. `combined_S` is `None` in that case.

## Three leaderboards

TT, TE, EE share a 2×2 Wishart covariance — they must be emulated jointly. φφ is independent. These are the natural composability units, so there are three ranked tracks:

1. **Combined track (primary).** `combined_S` computed from `mae_total` + timing. Requires a submission that emits all four spectra consistently (in particular, $|C_\ell^{TE}|^2 < C_\ell^{TT} \cdot C_\ell^{EE}$ at every $\ell$ so the 2×2 matrix stays positive definite).
2. **CMB-block track.** `combined_S` computed from `mae_cmb` + timing. Requires TT/TE/EE but not φφ.
3. **PP track.** `combined_S` from `mae_pp` + timing. Requires φφ only.

You can enter any subset of tracks. Two submissions can each win one of {CMB, PP} and a user could compose the two into a full emulator.

Per-spectrum raw-$C_\ell$ MAE (TT-only, TE-only, EE-only, PP-only) is exposed by other utilities but **not ranked** — those aren't physically valid likelihoods on their own.

## What to submit

For each track you enter:

1. The `Emulator` class (as above) and any additional Python modules needed to load and run it.
2. Trained weights / serialized state.
3. A training script that reproduces those weights from the released training set.
4. A short writeup (≤ 2 pages) covering:
   - Architecture and training hyperparameters.
   - Whether you used only the released training set or also regenerated more data. If the latter, a scaling curve (precision vs. $N_\text{train}$) is strongly encouraged.
   - Any pre/post-processing applied to the spectra (log, PCA, $D_\ell$ rescaling, etc.).

## Rules

### Allowed
- Any emulator architecture: MLPs, transformers, PCA + NN, Gaussian processes, symbolic regression.
- Any ML framework (JAX preferred for timing reasons, but PyTorch / TensorFlow / plain numpy also fine).
- Generating **additional** training cosmologies via `cmbemu.generate_data(...)` from the same parameter prior, and reporting how your architecture scales with $N_\text{train}$.
- Any amount of GPU compute for training.

### Prohibited
- Using the 5 000-point test set in any form of training or hyperparameter selection. It is held out for scoring only.
- Replacing the reference spectra with output from a different cosmology code (CLASS, CAMB, or other).
- Multi-threading inside `predict()` to inflate the timing benchmark; the container exposes 64 CPUs but the score is measured with JAX/TF pinned to a single one.

## Reference hardware (scoring container)

- 64 CPUs / 128 GB RAM / 1 GPU.
- **Scoring uses 1 CPU** because MCMC chains are single-threaded per chain.
- Training is done on your own hardware; any GPU is fine.

## Baselines

Two trivial baselines ship with the package so you can sanity-check your scoring setup:

```python
# Always returns the Planck fiducial spectrum, regardless of input
emu = cec.ConstantPlanck()

# 1-NN lookup on the training set in normalized parameter space
train = cec.load_train()
emu = cec.NearestNeighbour(train)

result = cec.get_score(emu)
```

Expected ballpark on the full test set: `ConstantPlanck` → `combined_S ≈ 7.05` (huge precision error; its microsecond speed hits the 1 ms floor so the timing term is 0). Any real emulator should beat this on precision by many decades.

## End-to-end example

A minimum viable submission, beginning to end:

```python
import numpy as np
import cmbemu as cec

class MyEmulator:
    def __init__(self):
        self.train = cec.load_train()  # your training data

    def predict(self, params: dict) -> dict[str, np.ndarray]:
        # replace with a trained model — this one just cheats
        # by returning the first training point
        idx = 0
        return {"tt": self.train["tt"][idx],
                "te": self.train["te"][idx],
                "ee": self.train["ee"][idx],
                "pp": self.train["pp"][idx]}

emu = MyEmulator()
result = cec.get_score(emu)
print(result)
```

Run this to confirm your install works end-to-end, then replace the body of `predict` with your actual model.
