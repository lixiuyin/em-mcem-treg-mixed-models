> **English** | [中文](README.zh.md)

# Student's t Regression: EM / MCEM Estimation (Thesis Chapter 3)

## Model

$$
y_i = x_i^{T}\beta + \varepsilon_i,\qquad \varepsilon_i \sim t_\nu(0,\sigma^2)\ \text{(scaled } t \text{ distribution)} .
$$

Introducing a latent scale variable $z_i\sim\mathrm{Gamma}(\nu/2,\nu/2)$ such that $\varepsilon_i\mid z_i\sim\mathcal N(0,\sigma^2/z_i)$, the marginal distribution is a $t$ with degrees of freedom $\nu$ and scale $\sigma^2$. The EM algorithm treats $z=(z_1,\dots,z_n)$ as the latent variables.

**Prior**: $\pi(\beta,\sigma^2)\propto 1/\sigma^2$ (thesis eq. 3-7), so the M-step divisor for $\sigma^2$ is $n+2$.

## Iteration Formulas

- **E-step weights**: $w_i=\mathbb E[z_i\mid y_i]=\dfrac{\nu+1}{\nu+r_i^2/\sigma^2}$, $r_i=y_i-x_i^{T}\beta$.
- **M-step $\beta$ (weighted least squares)**: $\hat\beta=(X^{T}WX)^{-1}X^{T}Wy$, $W=\mathrm{diag}(w_i)$.
- **M-step $\sigma^2$**: $\hat\sigma^2=\dfrac{1}{n+2}\sum_i w_i r_i^2$ (MAP under the $1/\sigma^2$ prior).
- **MCEM**: Draw $M$ samples from the posterior $z_i\mid y_i\sim\mathrm{Gamma}\!\big(\tfrac{\nu+1}{2},\,\tfrac{\nu+r_i^2/\sigma^2}{2}\big)$ and approximate $w_i$ by the sample mean.

## Files

| File | Description |
|---|---|
| `data_generation.R` | Generate simulated $t$-regression data; `random_init` (conservative) / `bold_random_init` (aggressive) initialisation |
| `utils.R` | EM/MCEM iteration (`run_single_em_or_mcem`), multiple-run simulation (`run_multiple_em_or_mcem`), MSE, plotting, degrees-of-freedom sensitivity study `study_df_effect` |
| `em.R` | EM examples: single run / multiple runs, initial-value sensitivity, degrees-of-freedom matching study |
| `mcem.R` | MCEM examples: comparison of sample sizes 1 vs 100 |
| `driver_compare500.R` | Comparability test of EM vs MCEM(M=100) over 500 simulations each (fixed seed 2025); outputs figures `21–24` |
| `driver_em_plots.R` | Single-run EM, conservative/aggressive initialisation, and degrees-of-freedom sweep figures (`25–30`) |
| `driver_mcem_plots.R` | Single-run MCEM iteration paths for `M=1` and `M=100` (`31–32`) |
| `driver_library_compare.R` | Side-by-side comparison of hand-coded EM/MCEM against `optim`/`hett::tlm`/`MASS::rlm`; outputs `library_compare_summary.txt` |
| `figures/` | Generated PNG figures |

## Default Parameters

`n=500, p=2, ν=10, σ²=0.4, β=(2,3,5)`, fixed starting values `β₀=(0,0,0), σ²₀=0.1`.

## Reproducing the Experiments

```bash
cd t-regression
Rscript driver_compare500.R      # comparison figures 21–24 + 500-run mean/MSE/timing
Rscript driver_em_plots.R        # figures 25–30: single-run EM, init strategies, df sweep
Rscript driver_mcem_plots.R      # figures 31–32: single-run MCEM (M=1, M=100)
# or run em.R / mcem.R interactively to view single-run and multi-run iteration plots
```

> The committed plots use English labels and base R fonts. If you customize them with Chinese labels and see boxes, register a local CJK font before running via `showtext`:
> `library(showtext); font_add("SimSun","<path>/SimSun.ttc"); showtext_auto(); showtext_opts(dpi=150)`.

## Validation Conclusions

- Both EM and MCEM converge to the maximum of the observed-data $t$ log-posterior under $\pi(\beta,\sigma^2)\propto1/\sigma^2$: $\hat\beta\approx(2.02,2.94,4.98)$, $\hat\sigma^2_{\text{MAP}}\approx0.42$. The fixed effects agree with the independent marginal $t$ MLE from `optim`, while the pure-MLE variance is $\hat\sigma^2_{\text{MLE}}\approx0.423$; the divisor difference $(n+2)$ vs. $n$ is negligible at $n=500$ (ratio $n/(n+2)\approx0.996$).
- Degrees-of-freedom matching study: $\sigma^2$ estimation accuracy is highest when the data's degrees of freedom match the algorithm's setting; convergence speed increases with degrees of freedom.

> Full derivation details are in thesis Chapter 3.
