# Emulator Competition — Status

## Settled decisions

### Scope
- **Single cosmology code: `classy_sz`** (via `classy_szfast`, `cosmo_model=6` = `ede-v2`). No CLASS, CAMB, or MontePython.
- Target: CMB TT, TE, EE, and $\phi\phi$ power spectra.
- Environment: `emucomp/` venv, Python 3.12.12, `cmbemu` installed editable.

### Parameter box

| Parameter    | Min    | Max    |
|--------------|:------:|:------:|
| ω_b          | 0.020  | 0.025  |
| ω_cdm        | 0.09   | 0.15   |
| H₀           | 55     | 85     |
| τ            | 0.03   | 0.10   |
| ln 10¹⁰A_s   | 2.7    | 3.3    |
| n_s          | 0.92   | 1.02   |

Covers Planck 2018 to ≥10σ per axis, comfortably inside the ede-v2 emulator training box. 64-corner probe passed.

### Datasets (frozen, uploaded)
- **50 000 train** + **5 000 test**, Latin-hypercube sampled, generated with `classy_szfast` at `cosmo_model=6`.
- float32 npz, $\ell \in [0, 6000]$ for TT/TE/EE and $[0, 3000]$ for $\phi\phi$.
- Plus **5 000 train_small + 500 test_small** dev subsets.
- **Hosted on Hugging Face** at [`borisbolliet/cmbemu-competition-v1`](https://huggingface.co/datasets/borisbolliet/cmbemu-competition-v1). Anyone with `pip install cmbemu` + `cec.load_test()` gets the data on first use.
- Seeds: train=202604, test=202605.

### Likelihood and scoring
- Exact **Wishart** log-likelihood at all $\ell \geq 2$, CV-limited ($f_\text{sky} = 1$).
- Primary precision score: **MAE on $\Delta\chi^2$**, all off-diagonal pairs, no cut.
- Sub-scores (natural composability units): **CMB block** (2×2 TT/TE/EE Wishart) and **PP block** (independent 1×1 Wishart).
- Speed: mean CPU wall time of 1 000 warm sequential `predict()` calls, CPU-pinned.
- Combined: $S = \log_{10}(\text{MAE}) + \alpha \log_{10}(\max(T_\text{CPU,ms}, T_\text{floor}))$ with $\alpha = 1$, $T_\text{floor} = 1$ ms (sub-ms inference buys no further credit).
- ConstantPlanck baseline: $\text{MAE} \approx 1.13 \times 10^{7}$, $T \approx 1.5\,\mu\text{s}$ → timing term floored to $0$, $S \approx 7.05$.

### Package (`cmbemu` v0.1.0a2)
- Installable via `pip install -e .` locally; PyPI publish is the next milestone.
- Public API: `load_train / load_test / download_data / generate_data / get_score / ConstantPlanck / NearestNeighbour`.
- CLI: `cmbemu download | generate | score-baseline`.
- All internal modules: box, lhc, generate, io, likelihood (with `chi2_matrix_split`), scoring (with `chi2_mae_blocks`), benchmark, data, baselines, api, cli, plotting.

## Done
- **`cmbemu` package scaffolded and installable.** `src/cmbemu/`, `pyproject.toml`, hatchling build.
- **HF dataset uploaded.** 4.55 GB across 5 files (README + 4 .npz); round-trip verified (cec.load_test from a clean cache produces bit-identical arrays to the local copy, and scoring ConstantPlanck on it returns the expected numbers).
- **Scoring implementation verified.** Matmul-vectorized Wishart chi^2 matches brute-force `np.linalg.inv` per-ℓ reference to ~10⁻¹³ relative error on random pairs; exactly 0 on diagonals.
- **Positive-definiteness confirmed.** $|TE|^2/(TT \cdot EE)$ max is 0.618 on train, 0.610 on test — comfortably below the Cauchy–Schwarz limit. `slogdet` never flips sign.
- **Plots at dpi=300**: LHC corner, spectra random/envelope/colored-by-param/dispersion, chi^2 PDF+CDF, score scatter. Produced for both `train` / `test` and their `_small` variants.
- **Docs**: `idea.md` (competition pitch + full Wishart / MAE / combined_S derivation — scoring.md was merged in), `data_description.md`, `CLAUDE.md`, `STATUS.md`, HF dataset README.

## What remains for v0.1 ship

1. **Publish `cmbemu` to PyPI.** `python -m build && twine upload`. (Needs a PyPI token; the wheel is already buildable.)
2. **GitHub repo.** `git init`, push to `borisbolliet/cmbemu`, wire up a CI that runs `cmbemu score-baseline --size small` as a smoke test.
3. **Ship precomputed `chi2_true`** inside `test.npz`? 5000×5000 float32 is 100 MB extra but saves ~2 s per `get_score`. Decide whether to version this.
4. **Competition container** spec + submission-upload mechanism (separate from this package).
5. **Example JAX emulator.** A small reference model to make the submission interface concrete for participants.

## Open questions deferred

- Should we report **Pareto-optimal composites** (best CMB-block × best PP) as a secondary leaderboard?
- How to handle **submitted emulator weights** size (Gb-scale NN checkpoints) — same HF org, a second dataset repo per submission?
- Do we want an **example training script** alongside the reference emulator, or just final weights + a short writeup?
