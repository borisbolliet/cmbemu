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

The score has two components, precision and speed, combined into a single
scalar.

### Setup

- **Test set**: $N = 5000$ parameter points $\{\theta_j\}_{j=1}^{N}$, each with
  a reference spectrum $C_\ell^{\mathrm{true}}(\theta_j)$ shipped with the
  data: TT, TE, EE for $\ell \in [2, 6000]$ and $\phi\phi$ for
  $\ell \in [2, 3000]$.
- Submission provides $C_\ell^{\mathrm{emu}}(\theta_j)$ for every test point.
- For each ordered pair $(i, j)$, treat $C_\ell^{\mathrm{true}}(\theta_i)$ as
  mock observed data $\hat D_i$ and the spectrum at $j$ (either true or
  emulated) as theory. Compute the full-sky Wishart log-likelihood.

### Per-pair $\chi^2$ (Wishart form)

At each multipole the spherical-harmonic coefficients $a_{\ell m}$ are
$(2\ell+1)$ i.i.d. Gaussian samples, so the empirical covariance is
Wishart-distributed. Taking $-2\log$ of the Wishart density and dropping
theory-independent constants gives the standard `mock_cmb`-style form:

$$
\chi^2(i, j) = \sum_{\ell=2}^{6000} (2\ell+1)\, f_{\mathrm{sky}}
\left[
\mathrm{Tr}\!\left( C_\ell(\theta_i)\, C_\ell(\theta_j)^{-1} \right)
+ \log \frac{|C_\ell(\theta_j)|}{|C_\ell(\theta_i)|}
- 2
\right]
+ \sum_{\ell=2}^{3000} (2\ell+1)\, f_{\mathrm{sky}}
\left[
\frac{C_\ell^{\phi\phi}(\theta_i)}{C_\ell^{\phi\phi}(\theta_j)}
+ \log \frac{C_\ell^{\phi\phi}(\theta_j)}{C_\ell^{\phi\phi}(\theta_i)}
- 1
\right]
$$

where

$$
C_\ell(\theta) = \begin{pmatrix} C_\ell^{TT}(\theta) & C_\ell^{TE}(\theta) \\ C_\ell^{TE}(\theta) & C_\ell^{EE}(\theta) \end{pmatrix}.
$$

This is the **exact** Wishart log-likelihood at all multipoles including
low-$\ell$; no Gaussian-in-$C_\ell$ approximation. The $(2\ell+1)f_{\mathrm{sky}}$
prefactor is the number of independent modes per multipole (correct
cosmic-variance weighting).

Defaults: $f_{\mathrm{sky}} = 1$ (cosmic-variance limited, no instrumental
noise), $\ell_{\max}^{\mathrm{CMB}} = 6000$, $\ell_{\max}^{\phi\phi} = 3000$.

Evaluating the formula with theory $= C^{\mathrm{true}}(\theta_j)$ gives
$\chi^2_{\mathrm{true}}(i, j)$; with theory $= C^{\mathrm{emu}}(\theta_j)$
gives $\chi^2_{\mathrm{emu}}(i, j)$. By construction
$\chi^2_{\mathrm{true}}(i, i) = 0$ exactly.

### Precision score: MAE on $\Delta\chi^2$

$$
S_{\mathrm{prec}} = \frac{1}{N(N - 1)} \sum_{i \neq j}
\bigl|\, \chi^2_{\mathrm{emu}}(i, j) - \chi^2_{\mathrm{true}}(i, j) \,\bigr|
$$

No clipping. All $N(N-1) = 24{,}995{,}000$ off-diagonal pairs count equally.
Diagonal pairs are excluded because $\chi^2_{\mathrm{true}}(i, i) = 0$ exactly.

**Why MAE and not MSE?** Under a fractionally accurate emulator,
$|\Delta\chi^2|$ grows roughly linearly with $\chi^2_{\mathrm{true}}$ (a
$\delta$-level spectral error gives $\Delta\chi^2 \sim 2\delta \cdot
\chi^2_{\mathrm{true}}$). Because the competition parameter box is wide
($\pm 20\sigma$ Planck on most axes) and the likelihood is
cosmic-variance-limited, $\chi^2_{\mathrm{true}}$ spans $10^4$–$10^8$ between
random test pairs. MSE would be quadratically tail-dominated — a handful of
extreme pairs would drown out everything else. MAE is linear in
$|\Delta\chi^2|$, preserves absolute-likelihood-level matching, and is
vastly less outlier-sensitive.

### Block decomposition

TT, TE, EE share a $2 \times 2$ Wishart covariance — they are emulated
together. $\phi\phi$ is independent (different likelihood, different signal).
So `get_score` returns three separate MAEs, computed with the same formula
restricted to the corresponding block:

| Key         | What it measures                                |
|-------------|-------------------------------------------------|
| `mae_cmb`   | TT/TE/EE $2\times 2$ Wishart block only        |
| `mae_pp`    | $\phi\phi$ $1\times 1$ Wishart block only      |
| `mae_total` | sum of both (the primary precision score)       |

These are the natural composability units for the three-track leaderboard
(see **Three leaderboards** below).

### Speed score

$T_{\mathrm{CPU}}$ = mean wall time (ms) per `predict(params_dict)` call.
Protocol:

- Sample `n_warmup + n_calls` uniform-random LHC points in the competition
  box (defaults 10 + 1000).
- Do `n_warmup` untimed warmup calls — absorbs JIT compile, lazy init,
  first-page-fault overhead.
- Time each of the remaining `n_calls` calls individually with
  `time.perf_counter_ns`. Only `emulator.predict(p)` is inside the timer;
  the dict conversion is outside.
- Report mean (used in `combined_S`), median, and standard deviation.
- Every call uses a **different** input to prevent result caching.
- **Single-threaded, CPU-only** — MCMC is sequential and typically runs on a
  CPU node. Export `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=` before
  importing your emulator. Pass `strict_cpu=True` to raise on GPU backends.

Mean, not median, because MCMC wall time is `mean × n_steps` — median would
underreport occasional slow calls.

### Combined score

$$
S = \log_{10}(S_{\mathrm{prec}}) + \alpha \cdot \log_{10}(\max(T_{\mathrm{CPU}},\ T_0))
$$

where $T_{\mathrm{CPU}}$ is `t_cpu_ms_mean` and $T_0$ is the soft floor on
the timing term. **Lower is better.** Defaults: $\alpha = 1$, $T_0 = 1$ ms.

- Above the floor, one decade of precision trades for one decade of speed
  (with $\alpha = 1$).
- **Below the floor, $T_0$ caps the benefit.** Sub-millisecond inference buys
  no further credit because realistic MCMC per-step overhead (proposal
  generation, other likelihood pieces, chain bookkeeping) dominates below
  $\sim 1$ ms. Same-precision emulators still break ties on speed, but only
  down to the floor.

### Defaults

| Symbol                        | Meaning                          |   Value |
|-------------------------------|----------------------------------|--------:|
| $N$                           | test-set size                    |    5000 |
| $\ell_{\max}^{\mathrm{CMB}}$  | TT/TE/EE upper $\ell$            |    6000 |
| $\ell_{\max}^{\phi\phi}$      | lensing upper $\ell$             |    3000 |
| $f_{\mathrm{sky}}$            | sky fraction                     |     1.0 |
| $\alpha$                      | precision-vs-speed weight        |     1.0 |
| $T_0$                         | soft floor on timing term (ms)   |     1.0 |

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

Computes both sub-scores above, then the **combined scalar** leaderboard number.
With $E$ = `mae_total`, $T$ = `t_cpu_ms_mean`, $T_0$ = `t_floor_ms`:

$$
S \;=\; \log_{10}(E) \,+\, \alpha \cdot \log_{10}(\max(T,\, T_0))
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

The same scalar can be recomposed from the two sub-scores. Calling
`get_score` is equivalent to running `get_accuracy_score` and
`get_time_score` and then this one line:

```python
acc = cec.get_accuracy_score(emu)
tim = cec.get_time_score(emu)
S   = cec.combined_score(acc["mae_total"]["mae"], tim["t_cpu_ms_mean"])
```

(Running both paths on the same model gives the same `S` up to timing
jitter across the two independent benchmark runs.)

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
