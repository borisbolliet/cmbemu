## Scoring

### Setup

- Test set: $N = 5000$ parameter points $\{\theta_j\}_{j=1}^{N}$, each with a
  true spectrum $C_\ell^{\text{true}}(\theta_j)$ from `classy_szfast`:
  TT, TE, EE for $\ell \in [2, 6000]$, $\phi\phi$ for $\ell \in [2, 3000]$.
- The submission provides, for every test $\theta_j$, an emulated spectrum
  $C_\ell^{\text{emu}}(\theta_j)$.
- For each ordered pair $(i, j)$ we treat the true spectrum at $i$ as mock
  data $\hat{D}_i$ and the spectrum at $j$ (either true or emulated) as
  theory, and compute the CV-limited, full-sky Wishart log-likelihood.

### Per-pair $\chi^2$ (Wishart form)

At each multipole $\ell$ the spherical-harmonic coefficients $a_{\ell m}$ are
$(2\ell+1)$ i.i.d. Gaussian samples, so the empirical covariance is
Wishart-distributed. Taking $-2\log$ of the Wishart density and dropping
theory-independent constants gives

$$
\chi^2(i, j) \;=\;
\sum_{\ell=2}^{\ell_{\max}^{\text{CMB}}} (2\ell+1)\, f_{\text{sky}}
\left[
\text{Tr}\!\left( C_\ell(\theta_i)\, C_\ell(\theta_j)^{-1} \right)
+ \log\frac{\lvert C_\ell(\theta_j) \rvert}{\lvert C_\ell(\theta_i) \rvert}
- 2
\right]
\;+\;
\sum_{\ell=2}^{\ell_{\max}^{\phi\phi}} (2\ell+1)\, f_{\text{sky}}
\left[
\frac{C_\ell^{\phi\phi}(\theta_i)}{C_\ell^{\phi\phi}(\theta_j)}
+ \log\frac{C_\ell^{\phi\phi}(\theta_j)}{C_\ell^{\phi\phi}(\theta_i)}
- 1
\right],
$$

where

$$
C_\ell(\theta) =
\begin{pmatrix}
C_\ell^{TT}(\theta) & C_\ell^{TE}(\theta) \\
C_\ell^{TE}(\theta) & C_\ell^{EE}(\theta)
\end{pmatrix}.
$$

This is the exact Wishart log-likelihood at all multipoles (including
low-$\ell$); no Gaussian-in-$C_\ell$ approximation. The $(2\ell+1)f_{\text{sky}}$
prefactor is the number of independent modes per multipole.

Evaluating the formula with theory $= C^{\text{true}}(\theta_j)$ gives
$\chi^2_{\text{true}}(i, j)$; with theory $= C^{\text{emu}}(\theta_j)$ gives
$\chi^2_{\text{emu}}(i, j)$. By construction $\chi^2_{\text{true}}(i, i) = 0$.

### Precision score (primary): MAE on $\Delta\chi^2$

$$
S_{\text{prec}} \;=\; \frac{1}{N(N - 1)} \sum_{i \neq j}
\bigl|\, \chi^2_{\text{emu}}(i, j) - \chi^2_{\text{true}}(i, j) \,\bigr|.
$$

No clipping. Uses all off-diagonal pairs. Diagonal pairs are excluded
because $\chi^2_{\text{true}}(i, i) = 0$ exactly; the emulator's error there is
a function of $|C^{\text{emu}}(\theta_i) - C^{\text{true}}(\theta_i)|$ alone,
which is reported as a separate diagnostic.

**Why MAE, not MSE?** Under a fractionally accurate emulator, $|\Delta\chi^2|$
grows roughly linearly with $\chi^2_{\text{true}}$ (a $\delta$-level spectral
error gives $\Delta\chi^2 \sim 2\delta\,\chi^2_{\text{true}}$). MSE would be
dominated quadratically by the tail of the $\chi^2_{\text{true}}$ distribution
— a handful of extreme pairs would drown out everything else. MAE is linear in
$|\Delta\chi^2|$ and preserves absolute-likelihood-level matching while
being vastly less outlier-sensitive.

### Speed score

$T_{\text{CPU}}$ is the mean wall time (in milliseconds) of a warm, sequential
call to the submission's `predict(params_dict)`, measured on the scoring
container with one CPU (JAX/TF pinned to CPU), over $\geq 1000$ calls with
parameters drawn fresh from the LHC at every call.

### Combined score

$$
S \;=\; \log_{10}\!\bigl(S_{\text{prec}}\bigr)
\;+\; \alpha \, \log_{10}\!\bigl(\max(T_{\text{CPU},\,\text{ms}},\; T_{\text{floor}})\bigr).
$$

Lower is better.

$\alpha$ is the precision-vs-speed tradeoff weight. With default $\alpha = 1$,
above the floor one decade of precision trades for one decade of speed.

$T_{\text{floor}} = 1$ ms is a **soft floor** on the timing term: sub-ms
inference does not buy further speed credit, because a realistic MCMC's
per-step overhead (proposal generation, other likelihood terms, chain
bookkeeping) dominates below ~1 ms and no further gains are available at the
chain level. Same-precision emulators still break ties on speed, but only
down to the floor.

### Defaults

| Symbol              | Meaning                     | Value |
|---------------------|-----------------------------|:-----:|
| $N$                 | test-set size               | 5000  |
| $\ell_{\max}^{\text{CMB}}$ | TT/TE/EE upper $\ell$ | 6000  |
| $\ell_{\max}^{\phi\phi}$   | lensing $\ell$         | 3000  |
| $f_{\text{sky}}$    | sky fraction                | 1.0   |
| $\alpha$            | speed weight                | 1.0   |
| $T_{\text{floor}}$  | soft floor on timing term (ms) | 1.0 |
