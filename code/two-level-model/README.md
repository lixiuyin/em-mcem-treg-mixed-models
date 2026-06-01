> **English** | [中文](README.zh.md)

# Two-Level Linear Model: EM / MCEM Estimation (Thesis Chapter 4)

## Model

$$
y_j = X_j\beta_j+\varepsilon_j,\qquad \beta_j=W_j\gamma+\mu_j,\qquad j=1,\dots,J,
$$
Substituting: $y_j = X_j W_j\gamma + X_j\mu_j+\varepsilon_j$, where $\mu_j\sim\mathcal N(0,D)$ and $\varepsilon_j\sim\mathcal N(0,\sigma^2 I_{n_j})$.
Parameters to estimate: fixed effect $\gamma$, second-level covariance $D$, and individual error variance $\sigma^2$. The latent variables are $\mu=(\mu_1,\dots,\mu_J)$.

**Prior convention (unified)**: flat non-informative prior $\pi(\gamma,D,\sigma^2)\propto 1$ ⇒ the M-step divisor for $D$ is $J$ (for MCEM: $M\!\cdot\!J$), and for $\sigma^2$ the divisor is the total sample size $N$.
Under this convention, `get_loglik` (the observed-data marginal log-likelihood) is the monotonically non-decreasing objective of EM.

## Iteration Formulas (corresponding to thesis equations)

- **E-step** (eq. 4-17/4-21): $\mu_{j}^{*}=(X_j^{T}X_j+\sigma^2 D^{-1})^{-1}X_j^{T}(y_j-X_jW_j\gamma)$, $V_j^{*}=\sigma^2(X_j^{T}X_j+\sigma^2 D^{-1})^{-1}$.
- **$\gamma$ update** (closed-form GLS, eq. 4-25b): $\hat\gamma=(\sum_j W_j^{T}\hat\Lambda_j^{-1}W_j)^{-1}\sum_j W_j^{T}\hat\Lambda_j^{-1}\hat\beta_j^{\text{OLS}}$, $\hat\Lambda_j=D+\sigma^2(X_j^{T}X_j)^{-1}$.
- **$D$ update** (eq. 4-32, ÷$J$): $\hat D=\frac{1}{J}\sum_j(\mu_j^{*}\mu_j^{*T}+V_j^{*})$.
- **$\sigma^2$ update** (eq. 4-35, ÷$N$): $\hat\sigma^2=\frac1N\sum_j[\lVert y_j-X_jW_j\gamma-X_j\mu_j^{*}\rVert^2+\mathrm{tr}(X_jV_j^{*}X_j^{T})]$.
- **MCEM**: $\gamma$ still uses closed-form GLS; $D=\frac{1}{MJ}\sum_l\sum_j\mu_j^{(l)}\mu_j^{(l)T}$, $\sigma^2=\frac{1}{MN}\sum_l\sum_j\lVert\cdot\rVert^2$.

## Files

| File | Description |
|---|---|
| `data_generation.R` | Generate two-level model data ($J$ groups, $n_j\in\{60,80,100\}$); `random_init`/`bold_random_init` |
| `utils.R` | Entry point: loads all split modules below; other scripts source only `utils.R` |
| `helpers.R` | Data accessors: `get_X_j`, `get_y_j`, `get_W_j`, `get_mu_j`, posterior helpers |
| `em_updates.R` | EM E-step/M-step: `em_update_gamma`, `em_update_D`, `em_update_sigma2` |
| `mcem_updates.R` | MCEM sampling and updates: `get_mu_samples`, `mcem_update_all` |
| `simulation.R` | `run_single_em_or_mcem`, `run_multiple_em_or_mcem` |
| `plotting.R` | All `plot_*` functions for iteration paths, MSE, zoomed views |
| `metrics.R` | `get_mse`, `get_loglik`, `print_result` |
| `em.R` | EM examples: single run, average of 500 runs, sensitivity to two types of initial values, comparison with MCEM |
| `mcem.R` | MCEM examples: sample sizes 10/100, single run and 500 runs |
| `driver_em_plots.R` | Generate figures `01–08` (thesis figures 4.1/4.2) |
| `driver_mcem_plots.R` | Generate figures `11–18` (thesis figures 4.3/4.4) |
| `driver_compare500.R` | EM vs MCEM(M=50) comparison over 500 runs; outputs figures `51–54` (fixed seed 2025, REML) |
| `driver_library_compare.R` | Side-by-side comparison of hand-coded EM/MCEM against `lme4` (both ML and REML); outputs `library_compare_summary.txt` |
| `figures/` | Generated PNG figures |

## Default Parameters

`J=20, p=2, q=3`, $n_j\in\{60,80,100\}$. True parameters per thesis §4.1.2 (drawn under `set.seed(2025)`): $D=Q\,\mathrm{diag}(5,6,7)\,Q^{\top}$ with a random orthogonal $Q$ (eigenvalues $(5,6,7)$), $\gamma_i\sim U(0,12)$, $\sigma^2\sim U(1,5)$ (drawn $\sigma^2\approx2.693$).
Fixed starting values $\hat\gamma_0=(0.1,\dots,0.1)$, $\hat D_0=\begin{psmallmatrix}4&2&1\\2&5&3\\1&3&6\end{psmallmatrix}$, $\hat\sigma^2_0=0.1$; tolerance $10^{-6}$ (plot drivers use max iterations 50; `driver_compare500.R` uses 30).

## Reproducing the Experiments

```bash
cd two-level-model
Rscript driver_em_plots.R       # figures 01–08
Rscript driver_mcem_plots.R     # figures 11–18 (includes M=100×500 runs, slow)
Rscript driver_compare500.R     # figures 51–54 + EM/MCEM numerical comparison
```

> The committed plots use English labels and base R fonts. If you customize them with Chinese labels, register a local CJK font using `showtext` as needed.

## Validation Conclusions (seed=2025, 500 simulations, `max_iter=30`)

The variance component $D$ uses **REML** estimation (`driver_compare500.R` sets `options(hlm_reml=TRUE)`, see main README §4.5.2):

- $\hat\gamma$ and $\hat\sigma^2$ accurately recover the true values; total MSE (average over 500 estimates then compute deviation norm): EM $\approx0.453$, MCEM(M=50) $\approx0.368$ (comparable order).
- $\hat D$ (REML) recovers the true $D$ well (Frobenius MSE $\approx0.28$), essentially unbiased even with the small group count $J=20$. Using ML instead (`options(hlm_reml=FALSE)`) shrinks $\hat D$ by a factor $(J-q-1)/J\approx0.8$ (every eigenvalue scaled by $\approx0.8$, i.e. about 20% underestimation) — the inherent downward bias of maximum-likelihood estimation for variance components (not a bug; `lme4` ML gives the same result). REML eliminates this bias by propagating the uncertainty in $\hat\gamma$.
- The observed/restricted log-likelihood increases monotonically; EM(ML)/EM(REML) agree digit-for-digit with `lme4`(ML)/`lme4`(REML) (see main README §6.2.1 and `driver_library_compare.R`).

> The thesis text (Chapter 4 / §6.2) has been reconciled with this code: flat prior ($\pi\propto1$, so $D$ divided by $J$), the QR-random true-parameter setup, and the refreshed table 4-2 values. See the root `README.md` (§4, §6.2) for the full derivations and results.
