> **English** | [中文](README.zh.md)

# General Linear Mixed Model (LMM): EM / MCEM Estimation

A generalisation of the two-level linear model: the fixed-effects design matrix $X$ and the random-effects design matrix $Z$ may differ from each other and can be specified arbitrarily.

## Model

$$
y_i = X_i\beta + Z_i u_i + \varepsilon_i,\qquad u_i\sim\mathcal N(0,G_0),\quad \varepsilon_i\sim\mathcal N(0,\sigma^2 I_{n_i}),\quad i=1,\dots,m,
$$

$X_i$ is an $n_i\times p$ fixed-effects design matrix (including an intercept); the random-effects design is $Z_i=X_i[\,,1{:}k]$ (random intercept plus random slopes for the first $k-1$ covariates).
Parameters to estimate: $\beta$ ($p$-dimensional), $G_0$ ($k\times k$ random-effects covariance), $\sigma^2$. The latent variables are the random effects $u$.

**Prior**: flat non-informative prior $\pi(\beta,G_0,\sigma^2)\propto 1$ ⇒ the ML M-step divisor for $G_0$ is the number of groups $m$ (for MCEM: $M\!\cdot\!m$), and for $\sigma^2$ the divisor is $n$.
Under this convention, `get_loglik` (the marginal log-likelihood) is the monotonically non-decreasing objective of EM.

For small or moderate group counts, ML variance components may be visibly downward-biased because fixed effects consume degrees of freedom. Set `options(lmm_reml=TRUE)` to use the REML correction: the code propagates the GLS fixed-effect uncertainty into both $G_0$ and $\sigma^2$ and switches `get_loglik` to the restricted log-likelihood.

The complete derivation is in **`lmm_derivation.md`** in this directory.

## Iteration Formulas

- **E-step**: $u_i^{*}=(Z_i^{T}Z_i+\sigma^2 G_0^{-1})^{-1}Z_i^{T}(y_i-X_i\beta)$, $V_i^{*}=\sigma^2(Z_i^{T}Z_i+\sigma^2 G_0^{-1})^{-1}$.
- **$\beta$ (closed-form GLS)**: $\hat\beta=(\sum_i X_i^{T}V_i^{-1}X_i)^{-1}\sum_i X_i^{T}V_i^{-1}y_i$, $V_i=Z_iG_0Z_i^{T}+\sigma^2 I$.
- **$G_0$ (ML ÷$m$)**: $\hat G_0=\frac1m\sum_i(u_i^{*}u_i^{*T}+V_i^{*})$.
- **$\sigma^2$ (ML ÷$n$)**: $\hat\sigma^2=\frac1n\sum_i[\lVert y_i-X_i\beta-Z_i u_i^{*}\rVert^2+\mathrm{tr}(Z_iV_i^{*}Z_i^{T})]$.
- **REML option**: adds fixed-effect uncertainty terms to the variance updates and uses the restricted log-likelihood; enabled by `options(lmm_reml=TRUE)`.
- **MCEM**: $\beta$ still uses closed-form GLS; $G_0=\frac{1}{mM}\sum_l\sum_i u_i^{(l)}u_i^{(l)T}$, $\sigma^2=\frac{1}{nM}\sum_l\sum_i\lVert\cdot\rVert^2$.

## Files

| File | Description |
|---|---|
| `data_generation.R` | `generate_lmm_data`, `random_init`/`bold_random_init` |
| `utils.R` | `get_post_u`, `em_update_beta/G0/sigma2`, `mcem_update_*`, `get_loglik`, `get_mse`, `run_single_em_or_mcem`, `run_multiple_em_or_mcem`, `plot_single_iteration` |
| `em.R` | EM examples: single run + average of 500 runs |
| `mcem.R` | MCEM examples: sample sizes 10/100, single run and multiple runs |
| `driver_compare500.R` | 500-simulation averaged figures `41/42` + paired EM vs MCEM ($M\in\{20,50,100,200\}$) comparison over 500 runs (fixed seed 2025; figure 42 is the MCEM $M=200$ averaged trajectory) |
| `driver_library_compare.R` | Side-by-side comparison of hand-coded EM/MCEM against `lme4`/`nlme::lme` under both ML and REML; outputs `library_compare_summary.txt` |
| `lmm_derivation.md` | Complete EM/MCEM derivation |
| `figures/` | Generated PNG figures |

## Default Parameters

`m=50` groups, $n_i\in\{15,20,25\}$, $\beta=(1,2,-1,0.5)$ ($p=4$), $G_0=\begin{psmallmatrix}4&1\\1&2\end{psmallmatrix}$ ($k=2$), $\sigma^2=1$;
fixed starting values $\beta_0=0$, $G_{0,0}=I_2$, $\sigma^2_0=0.5$.

## Reproducing the Experiments

```bash
cd linear-mixed-model
Rscript driver_compare500.R          # figures 41/42 + 500-run EM/MCEM numerical comparison
Rscript driver_library_compare.R     # ML/REML side-by-side validation against lme4 and nlme
# or run em.R / mcem.R interactively
```

> The committed plots use English labels and base R fonts. If you customize them with Chinese labels, register a local CJK font using `showtext` as needed.

## Validation Conclusions (seed=2025)

- **Single EM run**: converges in 10 steps with monotonically increasing observed-data log-likelihood; $\hat\beta$, $\hat G_0$ diagonal, and $\hat\sigma^2$ all approach the true values.
- **Average over 500 simulations**: $\hat\beta\approx(1.00,1.98,-1.00,0.50)$; $\hat G_0$ diagonal $\approx(3.92,1.95)$ (the off-diagonal, noisy on a single dataset, recovers to $\approx1$ on average; true value $1$); $\hat\sigma^2\approx0.995$; total MSE $\approx0.13$ — consistent with the §6.3 main table below.
- The slight underestimation of $\hat G_0$ diagonals is the inherent downward bias of ML for variance components; the off-diagonal has high variance in a single dataset but is accurately recovered after averaging over multiple runs.
- When the ML variance estimates are unsatisfactory, report REML side-by-side. `driver_library_compare.R` now verifies both `options(lmm_reml=FALSE)` and `options(lmm_reml=TRUE)` against `lme4`/`nlme` under the matching likelihood criterion.

### EM vs MCEM, 500 simulations (seed=2025, paired)

All configurations run on the **same** 500 datasets. Headline columns below; the full $M\in\{20,50,100,200\}$ sweep is in the top-level README §6.3.

| Metric | EM | MCEM(M=200) |
|---|---|---|
| $\hat\beta$ | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) |
| $\hat G_0$ diagonal | (3.921, 1.953) | (3.921, 1.952) |
| $\hat\sigma^2$ | 0.9950 | 0.9950 |
| Total MSE | 0.13292 | 0.13305 |
| Mean iterations | 8.34 | 50 (hit limit) |
| Total time | 30.6 s | 2839 s (≈93×) |

True values: $\beta=(1,2,-1,0.5)$, $G_0$ diagonal $=(4,2)$, $\sigma^2=1$. On identical data, $\hat\beta$ is **identical** (closed-form GLS) and MCEM reproduces EM at every $M$ (the Monte Carlo noise in the variance components shrinks with $M$); MCEM is much slower with no accuracy gain, because the E-step already has a closed form.
