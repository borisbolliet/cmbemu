# cmbemu — Emulator Competition

## Goal

Build neural-network emulators of the four CMB angular power spectra — **TT**, **TE**, **EE**, and lensing-potential **φφ** — that are simultaneously **precise** (matching `classy_szfast` ground truth to high accuracy) and **fast** (sub-millisecond `predict()` on CPU, to be usable inside MCMC chains). Both axes are scored; see [`scoring.md`](scoring.md) for the formal definition.

## Motivation

MCMC inference on CMB data is bottlenecked by Boltzmann solvers. Emulators deliver 10³–10⁶× speedups while preserving parameter-level accuracy, and have become the de facto tool for large cosmological likelihood codes (CosmoPowerJAX, classy_szfast, etc.). But the community lacks a common benchmark against which different architectures and training strategies can be compared apples-to-apples. This competition fills that gap:

- **Fixed dataset.** 50k training + 5k testing spectra drawn from the same Latin-hypercube, generated identically with `classy_szfast` (cosmo_model=6, ede-v2 stack in near-LCDM configuration). Pre-hosted on Hugging Face.
- **Fixed evaluation.** Pairwise Wishart $\chi^2$ MAE + single-CPU warm-call timing. No ambiguity about what "good" means.
- **Fixed hardware envelope.** Training on GPU; scoring on 1 CPU of the 64-CPU reference container.

## Parameter prior

Six LCDM parameters, Latin-hypercube sampled inside a box that covers Planck 2018 to ≥10σ on every axis while staying well inside the `classy_szfast` ede-v2 emulator validity region:

| Parameter        | Min    | Max    |
|------------------|:------:|:------:|
| ω_b              | 0.020  | 0.025  |
| ω_cdm            | 0.09   | 0.15   |
| H₀               | 55     | 85     |
| τ                | 0.03   | 0.10   |
| ln 10¹⁰ A_s      | 2.7    | 3.3    |
| n_s              | 0.92   | 1.02   |

## Scoring (summary)

For each ordered pair $(i, j)$ of test points, treat $C_\ell^{\text{true}}(\theta_i)$ as mock CV-limited data and compute the Wishart log-likelihood with theory = emulator or theory = ground truth. The primary precision score is the **MAE on $\Delta\chi^2$** across all off-diagonal pairs (no cut, absolute-likelihood-level matching).

Combined score:

$$
S \;=\; \log_{10}(\text{MAE on } \Delta\chi^2)
\;+\; \alpha \, \log_{10}(T_{\text{CPU, ms}}),\qquad \alpha = 1.
$$

**Lower is better.** One decade of precision trades for one decade of speed. See [`scoring.md`](scoring.md) for the full mathematical form.

## Three leaderboards

Because **TT/TE/EE share a 2×2 Wishart covariance** and **φφ is independent**, these are the natural composability units:

1. **Combined track (primary).** `mae_total` + speed → `combined_S`. Requires submitting all four spectra consistently.
2. **CMB-block track.** Only the TT/TE/EE contribution to $\chi^2$, scored with the 2×2 Wishart. Requires submitting TT/TE/EE jointly (positive-definite constraint).
3. **PP track.** Only the $\phi\phi$ contribution. Independent of the other three — can be submitted alone.

Participants can enter any subset of tracks. Per-spectrum raw-$C_\ell$ MAE (TT-only, TE-only, EE-only, PP-only) is reported as a diagnostic but is not ranked — those aren't physically valid likelihoods.

## Rules

### Allowed
- Any emulator architecture (transformers, MLPs, PCA + NN, Gaussian processes, symbolic regression...).
- Any ML framework (JAX preferred, but PyTorch / TensorFlow / plain numpy also fine).
- Generating **additional** training data via `cmbemu.generate_data(...)` — participants are encouraged to study how their architecture scales with $N_\text{train}$ and report this in their writeup.
- Mixing and matching emulators across tracks — e.g. someone might win CMB-block and someone else the PP track.

### Prohibited
- Using any cosmology code other than `classy_sz` / `classy_szfast` as ground truth.
- Training or evaluating against the 5k **test set** — it is held out for scoring only.
- Scoring under a non-default `cosmo_model`; the official dataset is locked to `cosmo_model=6`.

### Submission format
A Python object exposing

```python
class Emulator:
    def predict(self, params: dict) -> dict[str, np.ndarray]:
        """Given a dict with the six LCDM parameters (names as in
        cmbemu.PARAM_NAMES), return a dict with keys
        "tt", "te", "ee", "pp" and array shapes
        (LMAX_CMB + 1,) for tt/te/ee and (LMAX_PP + 1,) for pp.
        """
```

Submissions include: the emulator class + its trained weights, a training script (for reproducibility), and a short writeup discussing architecture, training setup, and $N_\text{train}$ scaling.

## Reference hardware

- **Training**: participants' choice (authors have an NVIDIA RTX 6000 Pro available).
- **Scoring container**: 64 CPUs / 128 GB / 1 GPU. The competition's `get_score` **times on 1 CPU** with JAX/TF pinned to CPU, because MCMC chains are single-threaded.

## Baselines shipped

To calibrate the leaderboard:

- **`ConstantPlanck`**: always returns the Planck fiducial spectrum. Ignores inputs. Zero-knowledge floor.
- **`NearestNeighbour`**: looks up the nearest training point in normalized parameter space. Zero interpolation, uses training data as a lookup table.

Both are importable as `cmbemu.ConstantPlanck` / `cmbemu.NearestNeighbour`.
