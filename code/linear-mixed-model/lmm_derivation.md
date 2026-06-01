# EM and MCEM Algorithms for the General Linear Mixed Model (Derivation)

> **English** | [中文](lmm_derivation.zh.md)

> This document provides a complete derivation of the EM and MCEM iterative updates for the general Linear Mixed Model (LMM).
> It follows the same Bayesian/EM framework and notation conventions as the two-level linear model in Chapter 4 of the paper
> (flat non-informative prior; the random-effects covariance $G_0$ M-step divided by the number of groups $m$, while $\sigma^2$ is divided by the total sample size $n$).
> The two-level linear model is a special case of this model.

## 1. Model Specification

For $i=1,\dots,m$ level-2 units (groups),

$$
y_i = X_i\beta + Z_i u_i + \varepsilon_i,\qquad
u_i \sim \mathcal N(0, G_0),\quad \varepsilon_i \sim \mathcal N(0,\sigma^2 I_{n_i}),
$$

where $X_i\in\mathbb R^{n_i\times p}$ is the fixed-effects design matrix (including an intercept column), $Z_i\in\mathbb R^{n_i\times k}$ is the random-effects design matrix,
$u_i$ (a $k$-dimensional random effect) and $\varepsilon_i$ are mutually independent, and groups are independent of one another.
This implementation sets $Z_i = X_i[\,,1{:}k]$, i.e., "random intercept plus random slopes for the first $k-1$ covariates" (a random-coefficient model).

Let $n=\sum_i n_i$. In stacked form: $y=X\beta+Zu+\varepsilon$, $u\sim\mathcal N(0, I_m\otimes G_0)$, $\varepsilon\sim\mathcal N(0,\sigma^2 I_n)$.
The parameters to be estimated are the fixed effects $\beta$, the random-effects covariance $G_0$ ($k\times k$), and the individual error variance $\sigma^2$.

**Relationship to the two-level linear model**: Setting $X_i=X_j W_j$, $Z_i=X_j$, $u_i=\mu_j$, $\beta=\gamma$, $G_0=D$ recovers the two-level model of Chapter 4. The present model is thus its generalization (fixed and random design matrices may differ and be specified independently).

## 2. Relevant Distributions

**(1) Marginal distribution of the observed data.** From $y_i = X_i\beta + Z_i u_i + \varepsilon_i$ and independence,

$$
y_i \sim \mathcal N\!\big(X_i\beta,\; V_i\big),\qquad V_i = Z_i G_0 Z_i^{T} + \sigma^2 I_{n_i}.
$$

**(2) Posterior of the latent variables.** Treating $u$ as latent variables, $(y,u)$ constitutes the complete data. Since groups are mutually independent,

$$
u_i \mid y_i \sim \mathcal N(u_i^{*}, V_i^{*}),\qquad
u_i^{*}=G_0 Z_i^{T}V_i^{-1}(y_i-X_i\beta),\quad
V_i^{*}=G_0 - G_0 Z_i^{T}V_i^{-1}Z_i G_0 .
$$

Applying the Woodbury matrix identity $V_i^{-1}=\sigma^{-2}I-\sigma^{-2}Z_i(\sigma^2G_0^{-1}+Z_i^{T}Z_i)^{-1}Z_i^{T}$, the expressions above reduce to computationally convenient forms:

$$
\boxed{\,u_i^{*}=\big(Z_i^{T}Z_i+\sigma^2 G_0^{-1}\big)^{-1}Z_i^{T}(y_i-X_i\beta),\qquad
V_i^{*}=\sigma^2\big(Z_i^{T}Z_i+\sigma^2 G_0^{-1}\big)^{-1}.}
$$

**(3) Complete-data conditional distribution.** $y_i\mid u_i \sim \mathcal N\!\big(X_i\beta+Z_i u_i,\ \sigma^2 I_{n_i}\big)$.

## 3. Prior and Complete-Data Posterior Log-Likelihood

We adopt the flat non-informative prior $\pi(\beta,G_0,\sigma^2)\propto 1$. Under this prior, the maximum a posteriori estimate coincides with the maximum likelihood estimate, so the EM algorithm effectively maximizes the marginal log-likelihood $\ln p(y\mid\beta,G_0,\sigma^2)$ of the observed data.

The complete-data posterior log-likelihood (retaining only terms that depend on the parameters) is:

$$
\ln L(\beta,G_0,\sigma^2\mid y,u)\propto
-\frac{n}{2}\ln\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\lVert y_i-X_i\beta-Z_i u_i\rVert^2
-\frac{m}{2}\ln\lvert G_0\rvert
-\frac{1}{2}\sum_{i=1}^{m}u_i^{T}G_0^{-1}u_i .
$$

## 4. EM Algorithm

**E-step (iteration $t$).** Using $\hat\beta_t,\hat G_{0,t},\hat\sigma^2_t$, compute the posterior $u_i\mid y_i\sim\mathcal N(u_{i,t}^{*},V_{i,t}^{*})$ and form the $Q$-function (using
$\mathbb E[u_i\mid y]=u_i^{*}$ and $\mathbb E[u_iu_i^{T}\mid y]=u_i^{*}u_i^{*T}+V_i^{*}$):

$$
\begin{aligned}
Q\propto&-\frac{n}{2}\ln\sigma^2
-\frac{1}{2\sigma^2}\sum_{i}\Big[\lVert y_i-X_i\beta-Z_i u_i^{*}\rVert^2+\mathrm{tr}(Z_i V_i^{*}Z_i^{T})\Big]\\
&-\frac{m}{2}\ln\lvert G_0\rvert-\frac{1}{2}\sum_i\Big[u_i^{*T}G_0^{-1}u_i^{*}+\mathrm{tr}(G_0^{-1}V_i^{*})\Big].
\end{aligned}
$$

**M-step.**

- **Update for $\beta$.** The strict EM update obtained by differentiating $Q$ with respect to $\beta$ gives $\hat\beta=\big(\sum_i X_i^{T}X_i\big)^{-1}\sum_i X_i^{T}(y_i-Z_i u_i^{*})$. This implementation instead uses the equivalent but faster-converging **closed-form GLS** (the marginal maximum likelihood estimate of $\beta$ given $G_0,\sigma^2$, i.e., an ECME step):

$$
\boxed{\;\hat\beta_{t+1}=\Big(\sum_{i=1}^{m}X_i^{T}V_i^{-1}X_i\Big)^{-1}\sum_{i=1}^{m}X_i^{T}V_i^{-1}y_i,\qquad V_i=Z_iG_0Z_i^{T}+\sigma^2 I_{n_i}.\;}
$$

Both approaches share the same convergence point (both maximize the marginal likelihood); the GLS update directly maximizes the marginal likelihood and avoids the slow convergence that can occur when variance components are large.

- **Update for $G_0$.** Differentiating $Q$ with respect to $G_0$ (using $\partial\ln|G_0|/\partial G_0=G_0^{-1}$ and $\partial(u^{T}G_0^{-1}u)/\partial G_0=-G_0^{-1}uu^{T}G_0^{-1}$) and setting the derivative to zero:

$$
m\,G_0=\sum_i\big(u_i^{*}u_i^{*T}+V_i^{*}\big)\ \Longrightarrow\
\boxed{\;\hat G_{0,t+1}=\frac{1}{m}\sum_{i=1}^{m}\big(u_i^{*}u_i^{*T}+V_i^{*}\big).\;}
$$

- **Update for $\sigma^2$.**

$$
\boxed{\;\hat\sigma^2_{t+1}=\frac{1}{n}\sum_{i=1}^{m}\Big[\lVert y_i-X_i\hat\beta_{t+1}-Z_i u_i^{*}\rVert^2+\mathrm{tr}(Z_i V_i^{*}Z_i^{T})\Big].\;}
$$

### 4.1 REML Correction for Variance Components

The ML formulas above use the correct ML divisors ($m$ for $G_0$, $n$ for $\sigma^2$), but they treat the GLS estimate $\hat\beta$ as fixed. With a limited number of groups this consumes fixed-effect degrees of freedom and can make variance components too small. REML corrects this by propagating the uncertainty of $\hat\beta$ into the second moments.

Per the ECM ordering (below), the two M-steps evaluate their REML pieces at **different** iterates — the $G_0$ update at the pre-update $G_{0,t}$, the $\sigma^2$ update at the just-updated $G_{0,t+1}$ — coinciding only at the fixed point. Write $V_{i,s}=Z_iG_{0,s}Z_i^T+\hat\sigma_t^2 I$ and

$$
C_s=\left(\sum_i X_i^T V_{i,s}^{-1}X_i\right)^{-1}.
$$

For $G_0$ (pieces at $G_{0,t}$), define $B_{i,t}=G_{0,t}Z_i^TV_{i,t}^{-1}X_i$. The REML update is

$$
\boxed{\;G_{0,t+1}^{\mathrm{REML}}=\frac1m\sum_i\left(u_i^*u_i^{*T}+V_i^*+B_{i,t}C_tB_{i,t}^T\right).\;}
$$

For $\sigma^2$ (pieces at $G_{0,t+1}$), define $A_{i,t+1}=X_i-Z_iG_{0,t+1}Z_i^TV_{i,t+1}^{-1}X_i$. The REML update is

$$
\boxed{\;\sigma_{t+1}^{2,\mathrm{REML}}=\frac1n\sum_i\left[
\lVert y_i-X_i\hat\beta_{t+1}-Z_i u_i^*\rVert^2+\mathrm{tr}(Z_iV_i^*Z_i^T)+\mathrm{tr}(A_{i,t+1}C_{t+1}A_{i,t+1}^T)\right].\;}
$$

This matches the code, where `get_reml_pieces` is called with the pre-update `G0` in `em_update_G0` and with the updated `G0_new` in `em_update_sigma2`.

The divisor remains $n$; the trace term supplies the fixed-effect degrees-of-freedom correction. In the special case with no random effects this fixed point reduces to the usual residual variance $RSS/(n-p)$. Under `options(lmm_reml=TRUE)`, `get_loglik` returns the restricted log-likelihood

$$
\ell_R=\ell_{\mathrm{ML}}(\hat\beta,G_0,\sigma^2)-\frac12\log\left|\sum_i X_i^TV_i^{-1}X_i\right|+\frac{p}{2}\log(2\pi).
$$

The implementation follows the ECM ordering: first update $\beta$ (closed-form GLS, independent of the posterior), then use the updated $(\hat\beta_{t+1},\hat G_{0,t},\hat\sigma^2_t)$ to compute the posterior and update $G_0$, then use the posterior under $(\hat\beta_{t+1},\hat G_{0,t+1},\hat\sigma^2_t)$ to update $\sigma^2$. Each sub-step does not decrease the relevant monotone objective: the observed-data marginal log-likelihood under ML, and the restricted log-likelihood $\ell_R$ (returned by `get_loglik`) under REML.

## 5. MCEM Algorithm

When the posterior cannot be integrated analytically (the present model still admits closed-form posteriors; MCEM is included here for demonstration and comparison), the E-step is replaced by a Monte Carlo approximation: draw $M$ samples $u_i^{(1)},\dots,u_i^{(M)}$ from $u_i\mid y_i\sim\mathcal N(u_i^{*},V_i^{*})$; then

$$
\hat G_{0,t+1}=\frac{1}{mM}\sum_{l=1}^{M}\sum_{i=1}^{m}u_i^{(l)}u_i^{(l)T},\qquad
\hat\sigma^2_{t+1}=\frac{1}{nM}\sum_{l=1}^{M}\sum_{i=1}^{m}\lVert y_i-X_i\hat\beta_{t+1}-Z_i u_i^{(l)}\rVert^2 .
$$

$\beta$ is still updated via closed-form GLS (analytically tractable; no sampling required). As $M\to\infty$, the MCEM updates converge to the corresponding EM updates by the law of large numbers.

Under `options(lmm_reml=TRUE)` the MCEM E-step mirrors the EM REML correction of §4.1: the $G_0$ samples are drawn from the inflated covariance $\mathcal N\!\big(u_i^{*},\,V_i^{*}+B_{i,t}C_tB_{i,t}^T\big)$ so the Monte Carlo second moment reproduces the REML $G_0$ update, while the $\sigma^2$ samples keep the ML covariance $\mathcal N(u_i^{*},V_i^{*})$ and the correction $\sum_i\mathrm{tr}(A_{i,t+1}C_{t+1}A_{i,t+1}^T)$ is added explicitly (avoiding double counting) — matching `get_u_samples` and `mcem_update_sigma2`.

## 6. Monotonicity and Convergence

Under the flat prior, the ML objective maximized by EM/ECME is the marginal log-likelihood of the observed data,
$\ln p(y\mid\beta,G_0,\sigma^2)=\sum_i\big[-\tfrac{n_i}{2}\ln 2\pi-\tfrac12\ln|V_i|-\tfrac12(y_i-X_i\beta)^{T}V_i^{-1}(y_i-X_i\beta)\big]$,
which is non-decreasing at every iteration. Under ML the function `get_loglik` computes this quantity (under REML it instead returns the restricted log-likelihood $\ell_R$ of §4.1, which is then the monotone objective); monotonicity has been verified in simulations with $m=50$ groups.
The maximum likelihood estimates of the variance components are known to be downward-biased (estimating fixed effects consumes degrees of freedom), so $\hat G_0$ and $\hat\sigma^2$ tend to be slightly underestimated. REML uses the restricted log-likelihood and the correction terms in §4.1 to remove this degrees-of-freedom bias; the code toggles this with `options(lmm_reml=TRUE)`.

## 7. Implementation Correspondence

| Formula | Code (`utils.R`) |
|---|---|
| Posterior $u_i^{*},V_i^{*}$ | `get_post_u` |
| $V_i$ | `get_V_i` |
| $\hat\beta$ (GLS) | `em_update_beta` |
| $\hat G_0$ (ML ÷$m$; REML adds $B_iCB_i^T$) | `em_update_G0` |
| $\hat\sigma^2$ (ML ÷$n$; REML adds $\mathrm{tr}(A_iCA_i^T)$) | `em_update_sigma2` |
| MCEM $\hat G_0,\hat\sigma^2$ | `mcem_update_G0`, `mcem_update_sigma2`, `mcem_update_all` |
| Marginal / restricted log-likelihood | `get_loglik` |
