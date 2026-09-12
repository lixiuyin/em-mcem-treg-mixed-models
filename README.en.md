# EM and MCEM Algorithms: Complete Derivations, Model Applications, and Empirical Results

> **English** | [中文 (Chinese)](README.md)

This repository documents the core content of an undergraduate thesis titled "Principles of the EM Algorithm and MCEM Algorithm and Their Applications to Parameter Estimation in Regression Models," providing complete derivations for EM, MCEM, Student's t regression, and the two-level / mixed linear models, together with the implementing code and empirical results.

To ensure stable GitHub rendering, standalone equations use `math` fenced blocks and inline equations uniformly use the `` $`...`$ `` form (backtick-wrapped, to avoid rendering failures when adjacent to punctuation). This document follows the final conventions of the revision:

- For t regression the prior $`\pi(\beta, \sigma^2)\propto 1/\sigma^2`$ is used, so the MAP update for $`\sigma^2`$ divides by $`n+2`$.

- For the two-level linear model the flat non-informative prior $`\pi(\gamma, D, \sigma^2)\propto 1`$ is used, so the EM update for $`D`$ divides by $`J`$ and the MCEM update divides by $`MJ`$.

- Fixed effects ($`\beta`$ / $`\gamma`$) are always updated via the closed-form marginal GLS (the ECME step); Monte Carlo approximation applies only to the variance components ($`D`$, $`\sigma^2`$).
- All model implementations follow the **multicycle-ECM** order: after each parameter block is updated, the E step is refreshed (conditional expectations are recomputed / samples are redrawn) before the next block is updated. This order shares the same fixed points as standard EM with a one-shot E step, and the marginal log-likelihood of the observed data remains monotonically non-decreasing (see §4.5, §4.6).
- The general linear mixed model (§5) is a generalization of the two-level model and uses the same conventions: flat prior, ML divisor for $`G_0`$ is the number of groups $`m`$ (MCEM: $`Mm`$), $`\sigma^2`$ is divided by total sample size $`n`$ (MCEM: $`Mn`$). When ML variance components are visibly downward-biased, `options(lmm_reml=TRUE)` enables REML and reports it side-by-side.

## Table of Contents

- [1. General EM Algorithm](#1-general-em-algorithm)
- [2. General MCEM Algorithm](#2-general-mcem-algorithm)
- [3. Student's t Regression Model](#3-students-t-regression-model)
- [4. Two-Level Linear Model](#4-two-level-linear-model)
- [5. General Linear Mixed Model](#5-general-linear-mixed-model)
- [6. Simulation Results](#6-simulation-results)
- [7. Empirical Analysis (Real Deep-Learning Data)](#7-empirical-analysis-real-deep-learning-data)
- [8. Repository Structure](#8-repository-structure)

## 1. General EM Algorithm

Let $`Y`$ denote the observed data, $`Z`$ the latent variables or missing data, and $`\theta`$ the parameters. The observed-data likelihood is

```math
L(\theta\mid Y)=p(Y\mid\theta)=\int p(Y,Z\mid\theta)\,dZ.
```

Directly maximizing this integral is generally intractable. The EM idea is to introduce the conditional distribution $`p(Z\mid Y,\theta_k)`$ under the current parameters $`\theta_k`$, and form the conditional expectation of the complete-data log-likelihood:

```math
Q(\theta\mid\theta_k)
=E_{\theta_k}[\log p(Y,Z\mid\theta)\mid Y].
```

### 1.1 Derivation via Jensen's Inequality

For any density $`q(Z)`$ we have

```math
\log p(Y\mid\theta)
=\log\int q(Z)\frac{p(Y,Z\mid\theta)}{q(Z)}\,dZ.
```

By Jensen's inequality,

```math
\log p(Y\mid\theta)
\geq
\int q(Z)\log\frac{p(Y,Z\mid\theta)}{q(Z)}\,dZ.
```

Setting $`q(Z)=p(Z\mid Y,\theta_k)`$ gives a lower bound on the observed log-likelihood:

```math
B(\theta\mid\theta_k)
=Q(\theta\mid\theta_k)
-E_{\theta_k}[\log p(Z\mid Y,\theta_k)\mid Y].
```

The second term does not depend on the $`\theta`$ being optimized, so maximizing $`B`$ is equivalent to maximizing $`Q`$.

<p align="center">
 <img src="code/general/figures/em_concept.png" width="60%" alt="EM lower-bound illustration">
</p>

### 1.2 EM Iteration Steps

E step:

```math
Q(\theta\mid\theta_k)
=E_{\theta_k}[\log p(Y,Z\mid\theta)\mid Y].
```

M step:

```math
\theta_{k+1}
=\arg\max_{\theta}Q(\theta\mid\theta_k).
```

If the M step only requires $`Q(\theta_{k+1}\mid\theta_k)\ge Q(\theta_k\mid\theta_k)`$, the result is the generalized EM (GEM).

### 1.3 Monotonicity

The observed log-likelihood can be decomposed as

```math
\log p(Y\mid\theta)
=Q(\theta\mid\theta_k)
-E_{\theta_k}[\log p(Z\mid Y,\theta)\mid Y].
```

Furthermore,

```math
\log p(Y\mid\theta)-\log p(Y\mid\theta_k)
=Q(\theta\mid\theta_k)-Q(\theta_k\mid\theta_k)
+KL(p(Z\mid Y,\theta_k)\,\|\,p(Z\mid Y,\theta)).
```

Because the KL divergence is non-negative, if the M step does not decrease $`Q`$, then

```math
\log p(Y\mid\theta_{k+1})
\ge
\log p(Y\mid\theta_k).
```

Therefore, the observed-data log-likelihood of EM is monotonically non-decreasing. The convergence points are typically stationary points of the likelihood; if the objective is unimodal, a unique maximum-likelihood estimate is obtained.

## 2. General MCEM Algorithm

When the conditional expectation in $`Q(\theta \mid \theta_k)`$ cannot be computed analytically, MCEM approximates the E step via Monte Carlo samples.

Draw samples from the current conditional distribution:

```math
Z^{(1)},\dots,Z^{(M)}
\sim p(Z\mid Y,\theta_k).
```

Approximate the $`Q`$ function by the sample average:

```math
Q_M(\theta\mid\theta_k)
=\frac{1}{M}\sum_{m=1}^{M}\log p(Y,Z^{(m)}\mid\theta).
```

The M step becomes

```math
\theta_{k+1}
=\arg\max_{\theta}Q_M(\theta\mid\theta_k).
```

When $`M`$ is large enough, the law of large numbers gives $`Q_M(\theta\mid\theta_k)\to Q(\theta\mid\theta_k)`$, so MCEM approximates EM. The cost is that sampling error introduces stochastic fluctuations in the iterative path, and the observed likelihood is no longer guaranteed to be strictly monotone at each step; it is common practice to increase $`M`$ with iterations or to accept approximate convergence at fixed $`M`$.

## 3. Student's t Regression Model

### 3.1 Model and Latent-Variable Representation

Let

```math
y_i=x_i^T\beta+\varepsilon_i,\qquad
\varepsilon_i\sim t_{\nu}(0,\sigma^2),\qquad i=1,\dots,n.
```

where $`x_i=(1,x_{i1},\dots,x_{ip})^T`$, $`\beta`$ is a $`p+1`$-dimensional regression coefficient vector, and the degrees of freedom $`\nu`$ is treated as given. In matrix form, stacking the $`n`$ observations as $`y=(y_1,\dots,y_n)^T`$ and $`X=(x_1,\dots,x_n)^T\in\mathbb{R}^{n\times(p+1)}`$,

```math
y=X\beta+\varepsilon,\qquad \varepsilon\sim t_\nu(0,\sigma^2 I_n)\ \text{(rows independent)}.
```

The t distribution can be represented as a normal-gamma mixture. Using the rate parameterization of the Gamma distribution:

```math
z_i\sim \mathrm{Gamma}(\frac{\nu}{2},\frac{\nu}{2}),
```

```math
y_i\mid z_i,\beta,\sigma^2
\sim N(x_i^T\beta,\frac{\sigma^2}{z_i}).
```

Thus $`z_i`$ serves as a latent variable. Conditioning on $`z_i`$, the model reduces to a weighted normal regression. Collecting the latent scales into the diagonal weight matrix $`W_z=\mathrm{diag}(z_1,\dots,z_n)`$, the conditional model is a heteroscedastic (precision-weighted) Gaussian,

```math
y\mid z,\beta,\sigma^2\sim N\!\big(X\beta,\ \sigma^2 W_z^{-1}\big),
```

so each $`z_i`$ scales the precision of row $`i`$. This is the matrix picture behind the EM: replacing the unobserved $`z_i`$ by their E-step expectations $`\gamma_i^{(k)}=E[z_i\mid y_i]`$ (§3.3) turns the M step into weighted least squares with weight matrix $`W_k=\mathrm{diag}(\gamma_1^{(k)},\dots,\gamma_n^{(k)})`$ (§3.5); a large residual drives $`z_i\downarrow`$, automatically down-weighting that observation — the origin of t-regression's robustness.

The correctness of this representation can be verified by marginalization: integrating out $`z_i`$,

```math
p(y_i\mid\beta,\sigma^2)
=\int_0^\infty
N\!\left(y_i;x_i^T\beta,\frac{\sigma^2}{z_i}\right)
\mathrm{Gamma}\!\left(z_i;\frac{\nu}{2},\frac{\nu}{2}\right)dz_i,
```

The integrand is the kernel of $`\mathrm{Gamma}\!\left(\frac{\nu+1}{2}, \frac{1}{2}\left[\nu+\frac{(y_i-x_i^T\beta)^2}{\sigma^2}\right]\right)`$ with respect to $`z_i`$; integrating yields the normalizing constant and exactly the Student's t density with degrees of freedom $`\nu`$ and scale $`\sigma^2`$, confirming that the marginal is $`t_\nu(x_i^T\beta,\sigma^2)`$.

### 3.2 Complete-Data Posterior Log-Likelihood

Under this mixture representation,

```math
p(y_i\mid z_i,\beta,\sigma^2)
=
\frac{z_i^{1/2}}{(2\pi\sigma^2)^{1/2}}
\exp[
-\frac{z_i}{2\sigma^2}(y_i-x_i^T\beta)^2
],
```

```math
p(z_i)
=
\frac{(\nu/2)^{\nu/2}}{\Gamma(\nu/2)}
z_i^{\nu/2-1}
\exp(-\frac{\nu z_i}{2}).
```

so the complete-data density satisfies

```math
p(y_i,z_i\mid\beta,\sigma^2)
=p(y_i\mid z_i,\beta,\sigma^2)p(z_i).
```

Expanding gives

```math
p(y_i,z_i\mid\beta,\sigma^2)
\propto
(\sigma^2)^{-1/2}
z_i^{(\nu+1)/2-1}
\exp[
-\frac{z_i}{2\sigma^2}(y_i-x_i^T\beta)^2
-\frac{\nu z_i}{2}
].
```

Taking the product over $`i=1,\dots,n`$ and retaining only terms involving $`\beta, \sigma^2`$:

```math
\log p(y,z\mid\beta,\sigma^2)
\propto
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}z_i(y_i-x_i^T\beta)^2.
```

The thesis uses the prior $`\pi(\beta,\sigma^2)\propto\frac{1}{\sigma^2}`$, so the complete-data posterior log-likelihood is

```math
\log \pi(\beta,\sigma^2\mid y,z)
\propto
-(\frac{n}{2}+1)\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}z_i(y_i-x_i^T\beta)^2.
```

### 3.3 Conditional Distribution of $`z_i\mid y_i`$

By Bayes' formula,

```math
p(z_i\mid y_i,\beta,\sigma^2)
=
\frac{p(y_i,z_i\mid\beta,\sigma^2)}{p(y_i\mid\beta,\sigma^2)}
\propto
z_i^{(\nu+1)/2-1}
\exp[
-\frac{z_i}{2}
(\nu+\frac{(y_i-x_i^T\beta)^2}{\sigma^2})
].
```

This is exactly the kernel of a Gamma distribution. Under the rate parameterization, i.e., if

```math
Z\sim \mathrm{Gamma}(a,b),
\qquad
p(z)\propto z^{a-1}\exp(-bz),
```

then here

```math
z_i\mid y_i,\beta,\sigma^2
\sim
\mathrm{Gamma}(
\frac{\nu+1}{2},
\frac{1}{2}[\nu+\frac{(y_i-x_i^T\beta)^2}{\sigma^2}]
).
```

Its conditional expectation is

```math
E[z_i\mid y_i,\beta,\sigma^2]
=
\frac{\nu+1}{\nu+(y_i-x_i^T\beta)^2/\sigma^2}.
```

At iteration $`k`$, define

```math
\gamma_i^{(k)}
=
\frac{\nu+1}{\nu+(y_i-x_i^T\beta_k)^2/\sigma_k^2}.
```

### 3.4 E Step of the t Regression EM

Substituting the conditional expectation gives

```math
Q(\beta,\sigma^2\mid\beta_k,\sigma_k^2)
\propto
-(\frac{n}{2}+1)\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}
\gamma_i^{(k)}(y_i-x_i^T\beta)^2.
```

Let $`W_k=\mathrm{diag}(\gamma_1^{(k)},\dots,\gamma_n^{(k)})`$.

### 3.5 M Step of the t Regression EM

Taking the partial derivative with respect to $`\beta`$:

```math
\frac{\partial Q}{\partial\beta}
=
\frac{1}{\sigma^2}
\sum_{i=1}^{n}\gamma_i^{(k)}x_i(y_i-x_i^T\beta)
=
\frac{1}{\sigma^2}
[
\sum_{i=1}^{n}\gamma_i^{(k)}x_i y_i
-
\sum_{i=1}^{n}\gamma_i^{(k)}x_i x_i^T\beta
].
```

Setting the partial derivative to zero:

```math
\sum_{i=1}^{n}\gamma_i^{(k)}x_i y_i
=
\sum_{i=1}^{n}\gamma_i^{(k)}x_i x_i^T\beta.
```

Therefore

```math
\beta_{k+1}
=
(X^T W_k X)^{-1}X^T W_k y.
```

Taking the partial derivative with respect to $`\sigma^2`$:

```math
\frac{\partial Q}{\partial\sigma^2}
=
-\frac{n/2+1}{\sigma^2}
+\frac{1}{2(\sigma^2)^2}
\sum_{i=1}^{n}\gamma_i^{(k)}(y_i-x_i^T\beta)^2.
```

Setting the partial derivative to zero:

```math
-(\frac{n}{2}+1)\sigma^2
+\frac{1}{2}
\sum_{i=1}^{n}\gamma_i^{(k)}(y_i-x_i^T\beta)^2
=0.
```

Rearranging gives

```math
(n+2)\sigma^2
=
\sum_{i=1}^{n}\gamma_i^{(k)}(y_i-x_i^T\beta)^2.
```

Substituting $`\beta_{k+1}`$:

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}
(y-X\beta_{k+1})^TW_k(y-X\beta_{k+1}).
```

This shows that the t regression EM is equivalent to iteratively reweighted least squares. Observations with larger residuals receive smaller weights, giving the model its robustness.

### 3.6 t Regression EM Algorithm

Given initial values $`\beta_0, \sigma_0^2`$, degrees of freedom $`\nu`$, maximum iterations $`K`$, and convergence threshold $`\varepsilon`$:

1. For $`k=0,1,2,\dots`$, repeat the E step and M step.
2. E step: compute the conditional weight for each observation:

```math
\gamma_i^{(k)}
=
\frac{\nu+1}{\nu+(y_i-x_i^T\beta_k)^2/\sigma_k^2},
\qquad i=1,\dots,n.
```

3. Let $`W_k=\mathrm{diag}(\gamma_1^{(k)},\dots,\gamma_n^{(k)})`$.

4. M step update:

```math
\beta_{k+1}=(X^TW_kX)^{-1}X^TW_ky.
```

5. Then update:

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}(y-X\beta_{k+1})^TW_k(y-X\beta_{k+1}).
```

6. If $`\max\!\big(\|\beta_{k+1}-\beta_k\|,\ \left|\sigma_{k+1}^2-\sigma_k^2\right|\big)\lt\varepsilon`$, stop; otherwise set $`k\leftarrow k+1`$ and continue.

### 3.7 t Regression MCEM

In MCEM, instead of using $`E[z_i\mid y_i]`$ directly, samples are drawn from the conditional distribution:

```math
z_i^{(k+1,m)}
\sim
\mathrm{Gamma}(
\frac{\nu+1}{2},
\frac{1}{2}[\nu+\frac{(y_i-x_i^T\beta_k)^2}{\sigma_k^2}]
),
\qquad m=1,\dots,M.
```

The conditional expectation is approximated by the sample mean:

```math
\bar z_i^{(k+1)}
=
\frac{1}{M}\sum_{m=1}^{M}z_i^{(k+1,m)}.
```

Let $`\bar W_k=\mathrm{diag}(\bar z_1^{(k+1)},\dots,\bar z_n^{(k+1)})`$. The approximate Q function is

```math
Q_M(\beta,\sigma^2\mid\beta_k,\sigma_k^2)
\propto
-(\frac{n}{2}+1)\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{i=1}^{n}\bar z_i^{(k+1)}(y_i-x_i^T\beta)^2.
```

The M step has the same form as in EM:

```math
\beta_{k+1}
=
(X^T\bar W_kX)^{-1}X^T\bar W_k y,
```

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}
(y-X\beta_{k+1})^T\bar W_k(y-X\beta_{k+1}).
```

### 3.8 t Regression MCEM Algorithm

Given initial values $`\beta_0, \sigma_0^2`$, degrees of freedom $`\nu`$, Monte Carlo sample size $`M`$, maximum iterations $`K`$, and threshold $`\varepsilon`$:

1. At each iteration $`k`$, first compute the current conditional distribution parameters:

```math
a_i^{(k)}=\frac{\nu+1}{2},
\qquad
b_i^{(k)}
=
\frac{1}{2}\left[\nu+\frac{(y_i-x_i^T\beta_k)^2}{\sigma_k^2}\right].
```

2. For each $`i`$, draw

```math
z_i^{(k+1,1)},\dots,z_i^{(k+1,M)}
\sim
\mathrm{Gamma}(a_i^{(k)},b_i^{(k)}).
```

3. Approximate the E step expectation by the sample mean:

```math
\bar z_i^{(k+1)}
=
\frac{1}{M}\sum_{m=1}^{M}z_i^{(k+1,m)}.
```

4. Form $`\bar W_k=\mathrm{diag}(\bar z_1^{(k+1)},\dots,\bar z_n^{(k+1)})`$.

5. Execute the M step, which has the same form as in EM:

```math
\beta_{k+1}=(X^T\bar W_kX)^{-1}X^T\bar W_ky,
```

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}(y-X\beta_{k+1})^T\bar W_k(y-X\beta_{k+1}).
```

6. Stop based on parameter change or maximum iterations. Because sampling error is present, the MCEM path may not be strictly monotone.

## 4. Two-Level Linear Model

### 4.1 Model Specification

Let group $`j`$ have $`n_j`$ observations:

```math
y_j=X_j\beta_j+\varepsilon_j,\qquad
\varepsilon_j\sim N(0,\sigma^2I_{n_j}),
```

The within-group regression coefficients satisfy the second-level model:

```math
\beta_j=W_j\gamma+\mu_j,\qquad
\mu_j\sim N(0,D),
```

where $`\gamma`$ are fixed effects, $`\mu_j`$ are random effects, and $`D`$ is the random-effects covariance matrix. Substituting:

```math
y_j=X_jW_j\gamma+X_j\mu_j+\varepsilon_j.
```

Define

```math
N=\sum_{j=1}^{J}n_j,\qquad
T=I_J\otimes D.
```

together with the block-diagonal design and the stacked second level — writing $`X_j\in\mathbb{R}^{n_j\times(p+1)}`$ ($`p`$ within-group covariates plus an intercept) and the group-level design $`W_j\in\mathbb{R}^{(p+1)\times(p+1)(q+1)}`$ (each of the $`p+1`$ coefficients regressed on $`q+1`$ group-level covariates, so $`\gamma\in\mathbb{R}^{(p+1)(q+1)}`$),

```math
X=\mathrm{diag}(X_1,\dots,X_J)\in\mathbb{R}^{N\times J(p+1)},\quad
W=\begin{pmatrix}W_1\\\vdots\\W_J\end{pmatrix}\in\mathbb{R}^{J(p+1)\times(p+1)(q+1)},\quad
\mu=\begin{pmatrix}\mu_1\\\vdots\\\mu_J\end{pmatrix}.
```

Stacking all groups gives

```math
y=XW\gamma+X\mu+\varepsilon,\qquad
\mu\sim N(0,T),\qquad
\varepsilon\sim N(0,\sigma^2I_N).
```

so the marginal distribution of the observed data is

```math
y\sim N(XW\gamma,V),\qquad
V=XTX^T+\sigma^2I_N.
```

Because $`X`$ and $`T`$ are block-diagonal, so is $`V=\mathrm{diag}(V_1,\dots,V_J)`$ with $`V_j=X_jDX_j^T+\sigma^2 I_{n_j}`$: the groups are independent, so every EM update factorizes group by group.

The joint distribution of the complete data has the covariance structure given in the revision:

```math
\begin{pmatrix} y \\ \mu \end{pmatrix}
\sim
N(
\begin{pmatrix}XW\gamma\\0\end{pmatrix},
\begin{pmatrix}
XTX^T+\sigma^2I_N & XT\\
TX^T & T
\end{pmatrix}
).
```

Note that the upper-right block is $`XT`$ and the lower-left block is $`TX^T`$.

### 4.2 Conditional Distribution of the Random Effects

By the conditional distribution formula for the multivariate normal:

```math
\mu\mid y,\gamma,D,\sigma^2
\sim N(\mu^*,V^*),
```

where

```math
\mu^*
=
TX^T(XTX^T+\sigma^2I_N)^{-1}(y-XW\gamma),
```

```math
V^*
=
T-TX^T(XTX^T+\sigma^2I_N)^{-1}XT.
```

Using the Woodbury identity, this can be written as

```math
\mu^*
=(X^TX+\sigma^2T^{-1})^{-1}X^T(y-XW\gamma),
```

```math
V^*
=
\sigma^2(X^TX+\sigma^2T^{-1})^{-1}.
```

The equivalence can be verified by the "push-through identity":

```math
(X^TX+\sigma^2T^{-1})^{-1}X^T
=
TX^T(XTX^T+\sigma^2I_N)^{-1},
```

Left-multiplying both sides by $`(X^TX+\sigma^2T^{-1})`$ and right-multiplying by $`(XTX^T+\sigma^2I_N)`$ both yield $`X^T(XTX^T+\sigma^2I_N)`$, confirming the identity and the alternative expression for $`\mu^*`$. $`V^*`$ is the right-hand side of the Woodbury identity

```math
(T^{-1}+\sigma^{-2}X^TX)^{-1}
=
T-TX^T(XTX^T+\sigma^2I_N)^{-1}XT
```

and $`(T^{-1}+\sigma^{-2}X^TX)^{-1}=\sigma^2(X^TX+\sigma^2T^{-1})^{-1}`$. Since the groups are independent, the conditional distribution for group $`j`$ is

```math
\mu_j\mid y_j,\gamma,D,\sigma^2
\sim N(m_j,V_j^*),
```

where

```math
m_j
=
(X_j^TX_j+\sigma^2D^{-1})^{-1}
X_j^T(y_j-X_jW_j\gamma),
```

```math
V_j^*
=
\sigma^2(X_j^TX_j+\sigma^2D^{-1})^{-1}.
```

### 4.3 Complete-Data Log-Likelihood

Using the flat prior $`\pi(\gamma,D,\sigma^2)\propto 1`$, MAP and MLE coincide, and EM maximizes the marginal observed-data log-likelihood $`\log p(y \mid \gamma, D, \sigma^2)`$. The parameter-dependent part of the complete-data log-likelihood is

```math
\ell_c(\gamma,D,\sigma^2\mid y,\mu)
\propto
-\frac{N}{2}\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma-X_j\mu_j\|^2
-\frac{J}{2}\log|D|
-\frac{1}{2}
\sum_{j=1}^{J}\mu_j^TD^{-1}\mu_j.
```

The coefficient of $`\log|D|`$ is $`-J/2`$, corresponding to the divisor $`J`$ in the $`D`$ update of the revision.

### 4.4 E Step of the Two-Level Model EM

At iteration $`k`$, using current estimates $`\gamma_k, D_k, \sigma_k^2`$, compute

```math
m_{jk}
=
(X_j^TX_j+\sigma_k^2D_k^{-1})^{-1}
X_j^T(y_j-X_jW_j\gamma_k),
```

```math
V_{jk}
=
\sigma_k^2(X_j^TX_j+\sigma_k^2D_k^{-1})^{-1}.
```

Two conditional expectations are needed:

```math
E[\mu_j\mid y,\theta_k]=m_{jk},
```

```math
E[\mu_j\mu_j^T\mid y,\theta_k]
=m_{jk}m_{jk}^T+V_{jk}.
```

Also,

```math
E[
\| y_j-X_jW_j\gamma-X_j\mu_j\|^2
\mid y,\theta_k
]
=
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T).
```

Therefore

```math
Q(\gamma,D,\sigma^2\mid\theta_k)
\propto
-\frac{N}{2}\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{j=1}^{J}
[
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T)
]
```

```math
-\frac{J}{2}\log|D|
-\frac{1}{2}
\sum_{j=1}^{J}
[
m_{jk}^TD^{-1}m_{jk}
+\mathrm{tr}(D^{-1}V_{jk})
].
```

### 4.5 M Step of the Two-Level Model EM

#### 4.5.1 Fixed Effects $`\gamma`$

Retaining the terms involving $`\gamma`$:

```math
\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2.
```

Taking the partial derivative with respect to $`\gamma`$ and setting it to zero:

```math
\sum_{j=1}^{J}
W_j^TX_j^T(y_j-X_jW_j\gamma-X_jm_{jk})
=0.
```

This gives the strict EM update:

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^TX_j^TX_jW_j)^{-1}
\sum_{j=1}^{J}W_j^TX_j^T(y_j-X_jm_{jk}).
```

The revision and code use the closed-form marginal GLS update as the actual update for $`\gamma`$. Let

```math
\hat\beta_j^{OLS}
=(X_j^TX_j)^{-1}X_j^Ty_j,
```

Then under the marginal model,

```math
\hat\beta_j^{OLS}
\sim
N(W_j\gamma,\Lambda_j),
\qquad
\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1}.
```

Given $`D_k, \sigma_k^2`$, the GLS update for $`\gamma`$ is

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS},
```

where $`\Lambda_{jk}=D_k+\sigma_k^2(X_j^TX_j)^{-1}`$. This formula is the conditional marginal MLE for $`\gamma`$ given $`D, \sigma^2`$; it agrees with the strict EM $`\gamma`$ update at convergence and substantially accelerates computation in the two-level model.

**Why the group-level GLS equals the full-sample marginal MLE.** Under the marginal model $`y_j \sim N(X_jW_j\gamma, V_j)`$ with $`V_j=X_jDX_j^T+\sigma^2 I_{n_j}`$, the full-sample GLS (i.e., marginal MLE) for $`\gamma`$ is

```math
\gamma
=
(\sum_{j}W_j^TX_j^TV_j^{-1}X_jW_j)^{-1}
\sum_{j}W_j^TX_j^TV_j^{-1}y_j.
```

Define

```math
\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1},
\qquad
H_j=X_j(X_j^TX_j)^{-1}X_j^T
```

A Woodbury expansion of $`V_j^{-1}`$ proves the two identities

```math
X_j^TV_j^{-1}X_j=\Lambda_j^{-1},
\qquad
X_j^TV_j^{-1}y_j=\Lambda_j^{-1}\hat\beta_j^{OLS},
```

the latter using the projection property $`X_j^TV_j^{-1}(I-H_j)=0`$ (since $`X_j^T(I-H_j)=0`$). Substituting immediately yields the GLS form in terms of $`\hat\beta_j^{OLS}`$. Thus "applying GLS to the group-level OLS estimates" is fully equivalent to "applying GLS to the individual observations," and both equal the conditional marginal MLE for $`\gamma`$ given $`(D,\sigma^2)`$; since EM as a whole converges to a stationary point of the marginal likelihood where the $`\gamma`$ component is that conditional MLE, both updates agree at convergence.

#### 4.5.2 Random-Effects Covariance $`D`$

Let $`S_{jk}=m_{jk}m_{jk}^T+V_{jk}`$. Retaining the terms involving $`D`$:

```math
-\frac{J}{2}\log|D|
-\frac{1}{2}\sum_{j=1}^{J}\mathrm{tr}(D^{-1}S_{jk}).
```

Using matrix calculus:

```math
\frac{\partial}{\partial D}\log|D|=D^{-1},
```

```math
\frac{\partial}{\partial D}\mathrm{tr}(D^{-1}S)
=-D^{-1}SD^{-1}.
```

Setting the partial derivative to zero:

```math
-\frac{J}{2}D^{-1}
+\frac{1}{2}\sum_{j=1}^{J}D^{-1}S_{jk}D^{-1}
=0.
```

Multiplying both sides by $`D`$ gives $`JD=\sum_{j=1}^{J}S_{jk}`$. Therefore (ML update)

```math
D_{k+1}^{\mathrm{ML}}
=
\frac{1}{J}
\sum_{j=1}^{J}
(m_{jk}m_{jk}^T+V_{jk}).
```

**Small-sample downward bias of ML and the REML correction.** The formula above treats $`\gamma`$ as known (substituting its estimate $`\hat\gamma`$). But since $`\hat\gamma`$ is estimated from the data, it "absorbs" part of the between-group variation, making the spread of $`m_{jk}`$ around $`W_j\hat\gamma`$ smaller and causing $`D_{k+1}^{\mathrm{ML}}`$ to systematically underestimate $`D`$. The magnitude is approximately $`(J-(q+1))/J`$ (the mean of each random-effect component is fitted by $`q+1`$ group-level coefficients); the bias is especially pronounced when the number of groups $`J`$ is small ($`J=20`$, $`q=3`$ gives a factor of $`\approx0.8`$, i.e., about 20% underestimation).

REML eliminates this bias by propagating the uncertainty in $`\hat\gamma`$ back into the random effects. Let the covariance of $`\hat\gamma`$ under marginal GLS be

```math
C=\Big(\sum_{j=1}^{J}W_j^T\Lambda_j^{-1}W_j\Big)^{-1},
\qquad
\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1},
```

From $`\hat\mu_j=DX_j^TV_j^{-1}(y_j-X_jW_j\hat\gamma)`$ we get $`\partial\hat\mu_j/\partial\hat\gamma=-DX_j^TV_j^{-1}X_jW_j=-D\Lambda_j^{-1}W_j=:-B_j`$ (using the identity $`X_j^TV_j^{-1}X_j=\Lambda_j^{-1}`$, see §4.5.1), so $`E[\mu_j\mu_j^T]`$ contains an additional term $`B_jCB_j^T`$. The REML update is

```math
D_{k+1}^{\mathrm{REML}}
=
\frac{1}{J}
\sum_{j=1}^{J}
\big(m_{jk}m_{jk}^T+V_{jk}+B_{jk}\,C_k\,B_{jk}^T\big),
\qquad
B_{jk}=D_k\Lambda_{jk}^{-1}W_j.
```

This update makes $`\hat D`$ approximately unbiased for any number of groups $`J`$; the corresponding monotonically non-decreasing objective changes from the observed-data marginal log-likelihood to the **restricted log-likelihood** $`\ell_R=\ell_{\mathrm{ML}}(\hat\gamma)+\tfrac12\log|C|+\tfrac{p_\gamma}{2}\log(2\pi)`$, $`p_\gamma=(p+1)(q+1)`$ (returned by `get_loglik` under REML; the additive $`\tfrac{p_\gamma}{2}\log(2\pi)`$ is the standard Harville constant from integrating out the fixed effects). Equivalently, using the REML projection matrix: $`D_{k+1}=\frac1J\sum_j\big(m_{jk}m_{jk}^T+D_k-D_kX_j^TP_{jj}X_jD_k\big)`$, where $`P=V^{-1}-V^{-1}(XW)C(XW)^TV^{-1}`$ and $`P_{jj}`$ is the $`n_j\times n_j`$ diagonal block of $`P`$ for group $`j`$.

> **Code switch.** The implementation toggles REML via the global option `options(hlm_reml=TRUE)` (enabled only in the simulation driver for this chapter); the default `FALSE` uses the ML update above. The empirical analysis of Chapter 5 (§7.2) keeps ML to match `lme4` ML results digit-for-digit. Under MCEM, samples are drawn from the inflated covariance $`N(m_{jk},\,V_{jk}+B_{jk}C_kB_{jk}^T)`$ so that the sample second moment matches $`D_{k+1}^{\mathrm{REML}}`$.

#### 4.5.3 Level-1 Error Variance $`\sigma^2`$

Retaining the terms involving $`\sigma^2`$:

```math
-\frac{N}{2}\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{j=1}^{J}
[
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T)
].
```

Define

```math
R_k(\gamma)
=
\sum_{j=1}^{J}
[
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T)
].
```

The relevant terms are $`-\frac{N}{2}\log\sigma^2-\frac{R_k(\gamma)}{2\sigma^2}`$. Taking the partial derivative with respect to $`\sigma^2`$:

```math
\frac{\partial Q}{\partial\sigma^2}
=
-\frac{N}{2\sigma^2}
+\frac{R_k(\gamma)}{2(\sigma^2)^2}.
```

Setting it to zero gives $`N\sigma^2=R_k(\gamma)`$. Substituting the updated $`\gamma_{k+1}`$:

```math
\sigma_{k+1}^2
=
\frac{1}{N}
\sum_{j=1}^{J}
[
\| y_j-X_jW_j\gamma_{k+1}-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T)
].
```

> **REML correction for $`\sigma^2`$.** The uncertainty in $`\hat\gamma`$ that inflates the $`D`$ update (§4.5.2) also enters the residual sum of squares. The group residual at the posterior mean, $`r_j=(I-X_jDX_j^TV_j^{-1})(y_j-X_jW_j\hat\gamma)`$, satisfies $`\partial r_j/\partial\hat\gamma=-A_j`$ with $`A_j=X_jW_j-X_jD\Lambda_j^{-1}W_j`$ (the same $`D`$ appears in both the standalone factor and in $`\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1}`$, since both arise from the single $`V_j=X_jDX_j^T+\sigma^2 I`$), so REML adds $`\sum_j\mathrm{tr}(A_jCA_j^T)`$ to $`R_k`$. **Unlike the $`D`$-update (§4.5.2), which evaluates its REML pieces at the pre-update $`D_k`$, this $`\sigma^2`$ block follows the multicycle-ECM refresh of §4.6 (step 4) and evaluates $`A_{jk}`$ and the GLS covariance at the just-updated $`D_{k+1}`$** — define $`\Lambda_{j,k+1}=D_{k+1}+\sigma_k^2(X_j^TX_j)^{-1}`$ and $`C_{k+1}=(\sum_{j=1}^{J}W_j^T\Lambda_{j,k+1}^{-1}W_j)^{-1}`$, so that
> ```math
> \sigma_{k+1}^{2,\mathrm{REML}}
> =\frac{1}{N}\Big(R_k(\gamma_{k+1})+\sum_{j=1}^{J}\mathrm{tr}(A_{jk}\,C_{k+1}\,A_{jk}^T)\Big),
> \qquad A_{jk}=X_jW_j-X_jD_{k+1}\Lambda_{j,k+1}^{-1}W_j.
> ```
> (This matches the code's `get_reml_pieces(data, D_{k+1}, \sigma_k^2)` in `em_update_sigma2`, where both the standalone $`D_{k+1}`$ factor and the $`\Lambda^{-1}`$/$`C`$ pieces use the updated $`D_{k+1}`$ uniformly; the two indices coincide only at the fixed point $`D_k=D_{k+1}`$.) This is the special case (fixed design $`X_jW_j`$, random design $`X_j`$) of the general LMM $`\sigma^2`$ REML correction in §5, and makes the hand-coded REML $`\hat\sigma^2`$ match `lme4(REML=TRUE)` to machine precision; under ML the term vanishes and the divisor stays $`N`$. Under MCEM it is added explicitly while the $`\sigma^2`$ samples keep the ML covariance, avoiding double counting.

### 4.6 Two-Level Model EM Algorithm

> **Relationship between the §4.5 derivation and the implementation (multicycle-ECM).** §4.5 presents the textbook EM derivation: construct $`Q`$ once at $`\theta_k`$ and then simultaneously maximize over $`(\gamma, D, \sigma^2)`$. This section (and the code `two-level-model/utils.R`) actually follows the **multicycle-ECM** (Meng & Rubin, 1993) order — $`\gamma`$ is solved analytically in one step via marginal GLS (the ECME step), and after each parameter-block update the E step is **refreshed**: use $`\gamma_{k+1}`$ to recompute conditional expectations before updating $`D`$, then use $`(\gamma_{k+1}, D_{k+1})`$ to recompute conditional expectations before updating $`\sigma^2`$. Each conditional maximization step does not decrease the marginal observed-data log-likelihood, so this order shares the same fixed points as standard EM. The procedure below corresponds exactly to the code implementation.

Given initial values $`\gamma_0, D_0, \sigma_0^2`$, maximum iterations $`K`$, and convergence threshold $`\varepsilon`$, repeat for $`k=0,1,2,\dots`$:

1. E step: for each group $`j=1,\dots,J`$, compute the conditional mean and conditional covariance of the random effects using the current $`(\gamma_k, D_k, \sigma_k^2)`$:

```math
m_{jk}
=
(X_j^TX_j+\sigma_k^2D_k^{-1})^{-1}
X_j^T(y_j-X_jW_j\gamma_k),
```

```math
V_{jk}
=
\sigma_k^2(X_j^TX_j+\sigma_k^2D_k^{-1})^{-1}.
```

2. M step — fixed effects (marginal GLS / ECME step, independent of the posterior). Let

```math
\hat\beta_j^{OLS}=(X_j^TX_j)^{-1}X_j^Ty_j,
\qquad
\Lambda_{jk}=D_k+\sigma_k^2(X_j^TX_j)^{-1},
```

and update

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS}.
```

3. Refresh E step and update $`D`$: use the new $`\gamma_{k+1}`$ (still with $`D_k, \sigma_k^2`$) to recompute the conditional mean:

```math
m'_{jk}
=
(X_j^TX_j+\sigma_k^2D_k^{-1})^{-1}
X_j^T(y_j-X_jW_j\gamma_{k+1}),
```

$`V_{jk}`$ does not depend on $`\gamma`$ and therefore remains unchanged, so

```math
D_{k+1}
=
\frac{1}{J}\sum_{j=1}^{J}(m'_{jk}m_{jk}^{\prime T}+V_{jk}).
```

4. Refresh E step again and update $`\sigma^2`$: use $`\gamma_{k+1}`$ together with the **already-updated $`D_{k+1}`$** ($`\sigma_k^2`$ unchanged) to recompute

```math
m''_{jk}
=
(X_j^TX_j+\sigma_k^2D_{k+1}^{-1})^{-1}
X_j^T(y_j-X_jW_j\gamma_{k+1}),
\qquad
V'_{jk}
=
\sigma_k^2(X_j^TX_j+\sigma_k^2D_{k+1}^{-1})^{-1},
```

so that

```math
\sigma_{k+1}^2
=
\frac{1}{N}
\sum_{j=1}^{J}
[
\|y_j-X_jW_j\gamma_{k+1}-X_jm''_{jk}\|^2
+\mathrm{tr}(X_jV'_{jk}X_j^T)
].
```

5. If the changes in $`\gamma, D, \sigma^2`$ all fall below the threshold $`\varepsilon`$, stop.

> Note: the "refresh" of the conditional moments in steps 3 and 4 is what characterizes multicycle-ECM. If instead the $`m_{jk}, V_{jk}`$ from step 1 (based on $`(\gamma_k, D_k)`$) were reused throughout (i.e., a one-shot E step as in §4.5), the algorithm degenerates to standard EM; both versions share the same fixed points.

### 4.7 Two-Level Model MCEM

In MCEM, samples are drawn from the conditional distribution given at iteration $`k`$:

```math
\mu_j^{(k+1,m)}
\sim
N(m_{jk},V_{jk}),
\qquad
m=1,\dots,M.
```

> As in §4.6, the actual implementation follows the **multicycle-ECM** order rather than drawing one sample set and reusing it: $`\gamma`$ is updated first by closed-form GLS, then $`\mu`$ is sampled at the refreshed mean $`m'_{jk}`$ (using $`\gamma_{k+1}`$) before the $`D`$ update and **resampled** at $`m''_{jk}`$ (using $`\gamma_{k+1}, D_{k+1}`$) before the $`\sigma^2`$ update; §4.8 states the exact order. The one-shot draw at $`m_{jk}`$ shown here shares the same fixed points and is used for notational simplicity.

The Monte Carlo approximation to the Q function is

```math
Q_M(\gamma,D,\sigma^2\mid\theta_k)
\propto
-\frac{N}{2}\log\sigma^2
-\frac{J}{2}\log|D|
```

```math
-\frac{1}{2M}
\sum_{m=1}^{M}\sum_{j=1}^{J}
[
\frac{1}{\sigma^2}
\| y_j-X_jW_j\gamma-X_j\mu_j^{(k+1,m)}\|^2
+(\mu_j^{(k+1,m)})^TD^{-1}\mu_j^{(k+1,m)}
].
```

#### 4.7.1 Update for $`\gamma`$

As in the revision, since the marginal M step for $`\gamma`$ can be solved analytically, MCEM still uses the closed-form GLS update:

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS},
```

```math
\Lambda_{jk}=D_k+\sigma_k^2(X_j^TX_j)^{-1}.
```

Therefore Monte Carlo error does not enter the analytical update for $`\gamma`$; it primarily affects $`D`$ and $`\sigma^2`$.

#### 4.7.2 Update for $`D`$

Retaining the terms involving $`D`$:

```math
-\frac{J}{2}\log|D|
-\frac{1}{2M}
\sum_{m=1}^{M}\sum_{j=1}^{J}
(\mu_j^{(k+1,m)})^TD^{-1}\mu_j^{(k+1,m)}.
```

Taking the derivative and setting it to zero:

```math
-\frac{J}{2}D^{-1}
+\frac{1}{2M}
\sum_{m=1}^{M}\sum_{j=1}^{J}
D^{-1}\mu_j^{(k+1,m)}(\mu_j^{(k+1,m)})^TD^{-1}
=0.
```

yielding

```math
D_{k+1}
=
\frac{1}{MJ}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\mu_j^{(k+1,m)}(\mu_j^{(k+1,m)})^T.
```

#### 4.7.3 Update for $`\sigma^2`$

Retaining the terms involving $`\sigma^2`$. Define

```math
R_M(\gamma)
=
\sum_{m=1}^{M}\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma-X_j\mu_j^{(k+1,m)}\|^2.
```

The relevant terms are $`-\frac{N}{2}\log\sigma^2-\frac{R_M(\gamma)}{2M\sigma^2}`$. Taking the partial derivative with respect to $`\sigma^2`$ and setting it to zero:

```math
-\frac{N}{2\sigma^2}
+\frac{R_M(\gamma)}{2M(\sigma^2)^2}
=0,
```

i.e., $`MN\sigma^2=R_M(\gamma)`$. Substituting $`\gamma_{k+1}`$:

```math
\sigma_{k+1}^2
=
\frac{1}{MN}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma_{k+1}-X_j\mu_j^{(k+1,m)}\|^2.
```

### 4.8 Two-Level Model MCEM Algorithm

The structure mirrors the EM procedure of §4.6; the implementation also follows the multicycle-ECM order — $`\gamma`$ is updated analytically first, and before each variance-component update the samples are **redrawn** using the latest parameters. Given initial values $`\gamma_0, D_0, \sigma_0^2`$, sample size $`M`$, maximum iterations $`K`$, and threshold $`\varepsilon`$, repeat for $`k=0,1,2,\dots`$:

1. M step — fixed effects (closed-form marginal GLS, no sampling needed):

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS}.
```

2. Under $`(\gamma_{k+1}, D_k, \sigma_k^2)`$, draw $`M`$ random-effect samples per group $`\mu_j^{(m)}\sim N(m'_{jk}, V_{jk})`$ ($`m'_{jk}`$ is the conditional mean from step 3 of §4.6), and update $`D`$:

```math
D_{k+1}
=
\frac{1}{MJ}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\mu_j^{(m)}(\mu_j^{(m)})^T.
```

3. Under $`(\gamma_{k+1}, D_{k+1}, \sigma_k^2)`$, **resample** $`\tilde\mu_j^{(m)}\sim N(m''_{jk}, V'_{jk})`$ ($`m''_{jk}, V'_{jk}`$ are the conditional moments based on $`D_{k+1}`$ from step 4 of §4.6), and update $`\sigma^2`$:

```math
\sigma_{k+1}^2
=
\frac{1}{MN}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\|y_j-X_jW_j\gamma_{k+1}-X_j\tilde\mu_j^{(m)}\|^2.
```

4. Since the E step is a stochastic approximation, the stopping criterion in practice typically combines maximum iterations, parameter change, and stability across multiple runs; sampling error means the path may not be strictly monotone (§2).

## 5. General Linear Mixed Model

The general linear mixed model (LMM) is a generalization of the two-level model: the fixed-effect design $`X_i`$ and the random-effect design $`Z_i`$ can differ and are specified freely. The two-level model is a special case (with fixed design $`X_jW_j`$, random design $`X_j`$, $`u_i=\mu_j`$, $`G_0=D`$). A complete derivation is also available in `code/linear-mixed-model/lmm_derivation.md`.

### 5.1 Model Specification

For $`i=1,\dots,m`$ level-2 units (groups),

```math
y_i = X_i\beta + Z_i u_i + \varepsilon_i,
\qquad
u_i\sim N(0,G_0),
\quad
\varepsilon_i\sim N(0,\sigma^2 I_{n_i}).
```

where $`X_i\in\mathbb{R}^{n_i\times p}`$ is the fixed-effect design (including an intercept column), $`Z_i\in\mathbb{R}^{n_i\times k}`$ is the random-effect design, $`u_i`$ (a $`k`$-dimensional random effect) and $`\varepsilon_i`$ are mutually independent and independent across groups. This implementation takes $`Z_i=X_i[,1:k]`$ (random intercept plus random slopes for the first $`k-1`$ covariates, i.e., a random-coefficient model). Let $`n=\sum_i n_i`$. The parameters to estimate are $`\beta`$ ($`p`$-dimensional), $`G_0`$ ($`k\times k`$), and $`\sigma^2`$; the latent variables are $`u`$.

In stacked matrix form,

```math
y=X\beta+Zu+\varepsilon,\qquad
X=\begin{pmatrix}X_1\\\vdots\\X_m\end{pmatrix},\quad
Z=\mathrm{diag}(Z_1,\dots,Z_m),\quad
u=\begin{pmatrix}u_1\\\vdots\\u_m\end{pmatrix}\sim N(0,G),\ \ G=I_m\otimes G_0,
```

with $`\varepsilon\sim N(0,\sigma^2 I_n)`$. The fixed-effect design $`X`$ stacks by rows while the random-effect design $`Z`$ is **block-diagonal** — each group's $`u_i`$ acts only within its own block — so the marginal covariance $`V=ZGZ^T+\sigma^2 I_n=\mathrm{diag}(V_1,\dots,V_m)`$ is block-diagonal and the groups are independent. This is exactly the two-level model of §4 under the identification $`X_i\leftrightarrow X_jW_j`$, $`Z_i\leftrightarrow X_j`$, $`u_i\leftrightarrow\mu_j`$, $`G_0\leftrightarrow D`$, $`\beta\leftrightarrow\gamma`$.

### 5.2 Marginal Distribution and Random-Effects Posterior

Marginal distribution of the observed data:

```math
y_i\sim N(X_i\beta, V_i),
\qquad
V_i=Z_iG_0Z_i^T+\sigma^2 I_{n_i}.
```

Treating $`u`$ as the latent variable, with groups independent, the random-effects posterior is $`u_i\mid y_i\sim N(u_i^*, V_i^*)`$, which simplifies to a computationally convenient form via the Woodbury identity:

```math
u_i^*=(Z_i^TZ_i+\sigma^2 G_0^{-1})^{-1}Z_i^T(y_i-X_i\beta),
\qquad
V_i^*=\sigma^2(Z_i^TZ_i+\sigma^2 G_0^{-1})^{-1}.
```

### 5.3 Complete-Data Log-Likelihood and Prior

Using the flat non-informative prior $`\pi(\beta, G_0, \sigma^2)\propto 1`$, MAP and MLE coincide, and EM maximizes the marginal observed-data log-likelihood $`\log p(y\mid\beta, G_0, \sigma^2)`$. The parameter-dependent part of the complete-data log-likelihood is

```math
\ell_c(\beta,G_0,\sigma^2\mid y,u)
\propto
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\|y_i-X_i\beta-Z_iu_i\|^2
-\frac{m}{2}\log|G_0|
-\frac{1}{2}\sum_{i=1}^{m}u_i^TG_0^{-1}u_i.
```

The coefficient of $`\log|G_0|`$ is $`-m/2`$, corresponding to the divisor $`m`$ in the $`G_0`$ M step.

### 5.4 EM E Step and M Step

The E step uses $`E[u_i\mid y]=u_i^*`$ and $`E[u_iu_i^T\mid y]=u_i^*u_i^{*T}+V_i^*`$, giving

```math
Q\propto
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i}[\|y_i-X_i\beta-Z_iu_i^*\|^2+\mathrm{tr}(Z_iV_i^*Z_i^T)]
-\frac{m}{2}\log|G_0|
-\frac{1}{2}\sum_{i}[u_i^{*T}G_0^{-1}u_i^*+\mathrm{tr}(G_0^{-1}V_i^*)].
```

M step:

- Fixed effects $`\beta`$ (marginal GLS / ECME; the strict EM solution $`(\sum_i X_i^TX_i)^{-1}\sum_i X_i^T(y_i-Z_iu_i^*)`$ shares the same convergence point):

```math
\beta_{t+1}=(\sum_{i=1}^{m}X_i^TV_i^{-1}X_i)^{-1}\sum_{i=1}^{m}X_i^TV_i^{-1}y_i,
\qquad
V_i=Z_iG_0Z_i^T+\sigma^2 I_{n_i}.
```

- Random-effects covariance $`G_0`$ (ML ÷$`m`$):

```math
G_{0,t+1}=\frac{1}{m}\sum_{i=1}^{m}(u_i^*u_i^{*T}+V_i^*).
```

- Error variance $`\sigma^2`$ (ML ÷$`n`$):

```math
\sigma^2_{t+1}=\frac{1}{n}\sum_{i=1}^{m}[\|y_i-X_i\beta_{t+1}-Z_iu_i^*\|^2+\mathrm{tr}(Z_iV_i^*Z_i^T)].
```

The implementation follows the same multicycle-ECM order as §4.6: first update $`\beta`$ (GLS, independent of the posterior), then use the posterior under $`(\beta_{t+1}, G_{0,t}, \sigma_t^2)`$ to update $`G_0`$, then use the posterior under $`(\beta_{t+1}, G_{0,t+1}, \sigma_t^2)`$ to update $`\sigma^2`$.

For REML, the implementation adds the fixed-effect uncertainty terms $`B_iCB_i^T`$ to the $`G_0`$ update and $`\mathrm{tr}(A_iCA_i^T)`$ to the $`\sigma^2`$ update, where $`B_i=G_0Z_i^TV_i^{-1}X_i`$, $`A_i=X_i-Z_iG_0Z_i^TV_i^{-1}X_i`$, and $`C=\big(\sum_{i=1}^{m}X_i^TV_i^{-1}X_i\big)^{-1}`$ is the marginal-GLS covariance of $`\hat\beta`$ (these are §4.5.2/§4.5.3 specialized to fixed design $`X_i`$, random design $`Z_i`$, covariance $`G_0`$); `get_loglik` returns the restricted log-likelihood $`\ell_R=\ell_{\mathrm{ML}}(\hat\beta)+\tfrac12\log|C|+\tfrac{p}{2}\log(2\pi)`$ with $`p=\dim\beta`$. This keeps the ML degrees of freedom explicit while providing an unbiased REML alternative for variance components.

### 5.5 MCEM

Draw $`M`$ samples $`u_i^{(1)},\dots,u_i^{(M)}`$ from $`u_i\mid y_i\sim N(u_i^*,V_i^*)`$ (also resampling with the latest parameters before each variance-component update); $`\beta`$ still uses the closed-form GLS; the variance components are replaced by sample approximations:

```math
G_{0,t+1}=\frac{1}{mM}\sum_{l=1}^{M}\sum_{i=1}^{m}u_i^{(l)}u_i^{(l)T},
\qquad
\sigma^2_{t+1}=\frac{1}{nM}\sum_{l=1}^{M}\sum_{i=1}^{m}\|y_i-X_i\beta_{t+1}-Z_iu_i^{(l)}\|^2.
```

As $`M\to\infty`$, each update converges to the corresponding EM update by the law of large numbers.

## 6. Simulation Results

### 6.1 t Regression Simulation Results

**Single-run EM: iteration path and MSE** (Fig. 3.1 / 3.2, left)

<p align="center">
 <img src="code/t-regression/figures/25_t_em_single_iter.png" width="45%" alt="t regression EM single-run iteration path">
 <img src="code/t-regression/figures/26_t_em_single_mse.png" width="45%" alt="t regression EM single-run MSE over iterations">
</p>

**EM averaged over 500 simulations: iteration path and MSE** (Fig. 3.1 / 3.2, right; Fig. 3.6)

<p align="center">
 <img src="code/t-regression/figures/21_t_em_500sim_iter.png" width="45%" alt="t regression EM averaged over 500 simulations: iteration path">
 <img src="code/t-regression/figures/22_t_em_500sim_mse.png" width="45%" alt="t regression EM 500 simulations: MSE over iterations">
</p>

**Initialization strategies over 500 simulations: conservative vs. wide-range** (Fig. 3.3)

<p align="center">
 <img src="code/t-regression/figures/27_t_em_normal_init_iter.png" width="45%" alt="t regression EM conservative-initialization iteration path (500 simulations)">
 <img src="code/t-regression/figures/28_t_em_bold_init_iter.png" width="45%" alt="t regression EM wide-range-initialization iteration path (500 simulations)">
</p>

**Degrees-of-freedom sweep: estimation error and iteration count** (Fig. 3.4)

<p align="center">
 <img src="code/t-regression/figures/29_t_df_error.png" width="45%" alt="t regression parameter estimation error vs degrees of freedom">
 <img src="code/t-regression/figures/30_t_df_iter.png" width="45%" alt="t regression EM iteration count vs degrees of freedom">
</p>

**Single-run MCEM: M=1 vs M=100 samples** (Fig. 3.5)

<p align="center">
 <img src="code/t-regression/figures/31_t_mcem_M1_single_iter.png" width="45%" alt="t regression MCEM(M=1) single-run iteration path">
 <img src="code/t-regression/figures/32_t_mcem_M100_single_iter.png" width="45%" alt="t regression MCEM(M=100) single-run iteration path">
</p>

**MCEM(M=100) averaged over 500 simulations: iteration path and MSE** (Fig. 3.6)

<p align="center">
 <img src="code/t-regression/figures/23_t_mcem_500sim_iter.png" width="45%" alt="t regression MCEM(M=100) averaged over 500 simulations: iteration path">
 <img src="code/t-regression/figures/24_t_mcem_500sim_mse.png" width="45%" alt="t regression MCEM(M=100) 500 simulations: MSE over iterations">
</p>

> Figures generated by `code/t-regression/driver_compare500.R` (21–24), `driver_em_plots.R` (25–30), and `driver_mcem_plots.R` (31–32), all with `set.seed(2025)` and saved in `code/t-regression/figures/`. The degrees-of-freedom sweep (29/30) uses a coarse df grid for runtime; the trend matches the thesis.

Average over 500 simulations ($`n=500`$, $`p=2`$, $`\nu=10`$, true values $`\beta=(2,3,5)`$, $`\sigma^2=0.4`$; fixed initial values $`\beta_0=0`$, $`\sigma_0^2=0.1`$):

| Metric | EM | MCEM $`M=100`$ |
|---|---:|---:|
| $`\hat\beta`$ | (2.0005, 3.0008, 5.0000) | (1.9975, 3.0000, 5.0014) |
| $`\hat\sigma^2`$ | 0.3977 | 0.3957 |
| MSE of $`\beta`$ | 8.62e−07 | 8.39e−06 |
| MSE of $`\sigma^2`$ | 5.50e−06 | 1.89e−05 |
| Total MSE | 6.36e−06 | 2.73e−05 |
| Mean iterations | 13.49 | 100.00 (reached limit) |
| Total time (s) | 11.0 | 336.7 |

Key findings:

- t regression EM is essentially iteratively reweighted least squares; observations with large residuals are automatically down-weighted.
- When the degrees of freedom specified in the algorithm match the true degrees of freedom in the data, $`\sigma^2`$ is estimated more accurately.
- $`\beta`$ is relatively insensitive to the degrees-of-freedom specification and remains stable overall.
- When $`M`$ is large enough, MCEM estimates approach analytic EM; when $`M`$ is small, the path is affected by sampling error.

#### 6.1.1 Comparison with Library Results

To independently verify the hand-coded implementation, `code/t-regression/driver_library_compare.R` (`set.seed(2025)`, $`n=500`$, $`\nu=10`$, 500 simulations) benchmarks the hand-coded EM/MCEM against three reference implementations: `optim` directly maximizing the marginal t likelihood (pure MLE), the third-party t regression library `hett::tlm`, and robust regression `MASS::rlm` (Huber M-estimation, different loss function).

**Single-dataset digit-by-digit comparison**: the hand-coded EM agrees digit-for-digit with "`optim` maximizing the same log-posterior (including the $`\pi\propto1/\sigma^2`$ prior)" ($`\max|\Delta\beta|=2\times10^{-8}`$, $`|\Delta\sigma^2|=4\times10^{-8}`$); $`\beta`$ agrees with `optim` (pure MLE) and `hett::tlm` to $`\approx10^{-5}`$; $`\mathrm{cor}(\beta_{\mathrm{EM}},\beta_{\mathrm{rlm}})=\mathrm{cor}(\beta_{\mathrm{EM}},\beta_{\mathrm{hett}})=1.0000`$.

**Mean comparison over 500 simulations** (true values $`\beta=(2,3,5)`$, $`\sigma^2=0.4`$):

| Method | Mean $`\hat\beta`$ | $`\hat\sigma^2`$ | Mean per-run MSE † | Note |
|---|---|---:|---:|---|
| Hand-coded EM (MAP) | (2.0005, 3.0008, 5.0000) | 0.3977 | 3.69e−03 | Prior $`\pi\propto1/\sigma^2`$, $`\sigma^2`$ divided by $`n+2`$ |
| `optim` (marginal MLE) | (2.0005, 3.0008, 5.0000) | 0.3997 | 3.69e−03 | Pure likelihood, $`\sigma^2`$ divided by $`n`$ |
| `hett::tlm` | (2.0005, 3.0008, 5.0000) | 0.3997 | 3.69e−03 | Third-party t regression MLE library |
| `MASS::rlm` | (2.0006, 3.0008, 5.0001) | — | 2.82e−03 | Huber M-estimation ($`\beta`$ only) |

> † **MSE definition note.** The "mean per-run MSE" in this table is the average over 500 simulations of the squared error in **each individual run** ($`\frac{1}{S}\sum_s[\,\lVert\hat\beta_s-\beta\rVert^2+(\hat\sigma^2_s-\sigma^2)^2\,]`$), measuring single-run estimation accuracy; the "total MSE" (6.36e−06) in the main table of §6.1 is computed by **first averaging the 500 estimates and then computing the deviation** ($`\lVert\bar{\hat\beta}-\beta\rVert^2+\dots`$), measuring estimator bias. These two quantities measure different things and cannot be compared directly; the hand-coded implementation agrees with all libraries under both definitions.

> **On the $`\sigma^2`$ discrepancy.** The hand-coded EM's $`\hat\sigma^2=0.3977`$ is slightly below `optim`/`hett`'s $`0.3997`$; the gap is exactly the MAP-to-MLE ratio $`\approx1-2/n`$ (dividing by $`n+2`$ vs. $`n`$), **not an implementation error**; $`\beta`$ is unaffected by the prior and agrees digit-for-digit across all three. This confirms that the derivation in §3.2 and §3.5 — "the prior $`\pi\propto1/\sigma^2`$ converts ML to MAP, changing the divisor from $`n`$ to $`n+2`$" — is self-consistent and correct.

### 6.2 Two-Level Linear Model Simulation Results

Setup: $`J=20`$ groups, $`p=2`$, $`q=3`$, within-group sample sizes $`n_j\in\{60,80,100\}`$. The true parameters follow the thesis §4.1.2 setup and are drawn under `set.seed(2025)`: $`D=Q\Lambda Q^{\top}`$ with a random orthogonal factor $`Q`$ (the QR factor of a Gaussian matrix) and fixed spectrum $`\Lambda=\mathrm{diag}(5,6,7)`$ — i.e. a random rotation with eigenvalues $`(5,6,7)`$; the fixed effects $`\gamma`$ and the level-1 error variance $`\sigma^2`$ are drawn from uniform distributions ($`\gamma_i\sim U(0,12)`$, $`\sigma^2\sim U(1,5)`$). Fixed initial values $`\gamma_0=0.1`$, $`\sigma_0^2=0.1`$. The variance component $`D`$ is estimated by **REML** (`options(hlm_reml=TRUE)`, see §4.5.2). The figures below are generated by `driver_em_plots.R`, `driver_mcem_plots.R`, and `driver_compare500.R` (all with `set.seed(2025)`) and saved in `code/two-level-model/figures/`. (The library validation further below uses a fixed $`D`$ with diagonal $`(5,6,7)`$ for a controlled comparison against `lme4`.)

> **Why REML.** The maximum-likelihood (ML) estimate of $`D`$ has an intrinsic downward bias of approximately $`(J-(q+1))/J`$ with few groups: at $`J=20`$, $`q=3`$, the factor is $`\approx0.8`$, causing the diagonal of $`\hat D`$ to be systematically about 20% below the truth (while $`\hat\sigma^2`$ is unaffected; this is not an implementation error — the ML estimator is consistent and §7.2 agrees with `lme4` ML digit-for-digit). Switching to REML (which propagates the uncertainty in $`\hat\gamma`$ back into the random effects by adding the correction term $`B_jCB_j^T`$ per group; derivation in §4.5.2) makes $`\hat D`$ approximately unbiased even at $`J=20`$ (diagonal recovering to approximately 100% of the truth). The empirical analysis in §7.2 retains ML to compare with `lme4` ML.

**Single-run EM iteration path and MSE**

<p align="center">
 <img src="code/two-level-model/figures/01_em_single_iter.png" width="45%" alt="Single-run EM iteration path">
 <img src="code/two-level-model/figures/02_em_single_mse.png" width="45%" alt="Single-run EM MSE over iterations">
</p>

**EM averaged over 500 simulations**

<p align="center">
 <img src="code/two-level-model/figures/03_em_500sim_iter.png" width="45%" alt="EM averaged over 500 simulations: iteration path">
 <img src="code/two-level-model/figures/04_em_500sim_mse.png" width="45%" alt="EM averaged over 500 simulations: MSE">
</p>

**Robustness of initialization strategies (top: same-distribution; bottom: aggressive)**

<p align="center">
 <img src="code/two-level-model/figures/05_em_normal_init_iter.png" width="45%" alt="Same-distribution initialization: iteration path">
 <img src="code/two-level-model/figures/06_em_normal_init_mse.png" width="45%" alt="Same-distribution initialization: MSE">
</p>
<p align="center">
 <img src="code/two-level-model/figures/07_em_bold_init_iter.png" width="45%" alt="Aggressive initialization: iteration path">
 <img src="code/two-level-model/figures/08_em_bold_init_mse.png" width="45%" alt="Aggressive initialization: MSE">
</p>

**Single-run MCEM iteration path and MSE (top: M=10; bottom: M=100)**

<p align="center">
 <img src="code/two-level-model/figures/11_mcem_M10_single_iter.png" width="45%" alt="Single-run MCEM(M=10) iteration path">
 <img src="code/two-level-model/figures/12_mcem_M10_single_mse.png" width="45%" alt="Single-run MCEM(M=10) MSE">
</p>
<p align="center">
 <img src="code/two-level-model/figures/13_mcem_M100_single_iter.png" width="45%" alt="Single-run MCEM(M=100) iteration path">
 <img src="code/two-level-model/figures/14_mcem_M100_single_mse.png" width="45%" alt="Single-run MCEM(M=100) MSE">
</p>

**MCEM averaged over 500 simulations (top: M=10; bottom: M=100)**

<p align="center">
 <img src="code/two-level-model/figures/15_mcem_M10_500sim_iter.png" width="45%" alt="MCEM(M=10) averaged over 500 simulations: iteration path">
 <img src="code/two-level-model/figures/16_mcem_M10_500sim_mse.png" width="45%" alt="MCEM(M=10) averaged over 500 simulations: MSE">
</p>
<p align="center">
 <img src="code/two-level-model/figures/17_mcem_M100_500sim_iter.png" width="45%" alt="MCEM(M=100) averaged over 500 simulations: iteration path">
 <img src="code/two-level-model/figures/18_mcem_M100_500sim_mse.png" width="45%" alt="MCEM(M=100) averaged over 500 simulations: MSE">
</p>

**EM vs. MCEM(M=50) comparison over 500 simulations**

<p align="center">
 <img src="code/two-level-model/figures/51_hlm_em_500sim_iter.png" width="45%" alt="Two-level model EM averaged over 500 simulations: iteration path">
 <img src="code/two-level-model/figures/52_hlm_em_500sim_mse.png" width="45%" alt="Two-level model EM 500 simulations: MSE">
</p>
<p align="center">
 <img src="code/two-level-model/figures/53_hlm_mcem_500sim_iter.png" width="45%" alt="Two-level model MCEM(M=50) averaged over 500 simulations: iteration path">
 <img src="code/two-level-model/figures/54_hlm_mcem_500sim_mse.png" width="45%" alt="Two-level model MCEM(M=50) 500 simulations: MSE">
</p>

Average over 500 simulations ($`J=20`$, REML, `max_iter=30`, output by `driver_compare500.R`; the true $`D`$ has eigenvalues $`(5,6,7)`$, the drawn $`\sigma^2\approx2.693`$). MSE of $`D`$ is the Frobenius error $`\lVert\hat D-D\rVert_F`$ (rotation-invariant):

| Metric | EM | MCEM $`M=50`$ |
|---|---:|---:|
| $`\hat\sigma^2`$ (truth $`\approx2.693`$) | 2.6891 | 2.6900 |
| MSE of $`\gamma`$ | 0.16810 | 0.16562 |
| MSE of $`D`$ | 0.28075 | 0.19965 |
| MSE of $`\sigma^2`$ | 0.00369 | 0.00282 |
| Total MSE | 0.45254 | 0.36808 |
| Mean iterations | 11.58 | 30.00 (reached limit) |
| Total time (s) | 61.1 | 320.1 |

> Reference: under the same setup with ML (`options(hlm_reml=FALSE)`), the eigenvalues of $`\hat D`$ are approximately $`(4.0,4.8,5.7)\approx0.8\times(5,6,7)`$ and MSE of $`D`$ is $`\approx2.2`$ — i.e., the old result with about 20% underestimation; REML reduces the MSE of $`D`$ to $`\approx0.21`$ (roughly 10-fold improvement).

Key findings:

- With REML, EM monotonically increases the **restricted log-likelihood** (§4.5.2); $`\gamma`$, $`\sigma^2`$, and $`D`$ all accurately recover the true values.
- At the small group count of $`J=20`$, the eigenvalue spectrum of $`\hat D`$ recovers to approximately 100% of the truth (ML would underestimate by about 20%), and $`D`$ no longer dominates total error — REML propagates the uncertainty in $`\hat\gamma`$ back to the random effects, eliminating the small-sample downward bias in variance components.
- EM and MCEM ($`M=50`$) achieve comparable accuracy; neither variance-component estimate dominates and both are close to the true values.
- MCEM often hits the maximum iteration count under strict tolerances (30 steps in this setup) due to Monte Carlo noise, and resampling at each step makes it substantially slower ($`\approx5.2\times`$), but parameter estimates remain close to EM.

#### 6.2.1 Comparison with Library Results

`code/two-level-model/driver_library_compare.R` (`set.seed(2025)`, $`J=20`$, 200 simulations) benchmarks the hand-coded EM/MCEM against the gold standard `lme4::lmer` on the same data, with **each estimand aligned separately**: hand-coded EM(ML) ↔ `lmer(REML=FALSE)`, hand-coded EM(REML) ↔ `lmer(REML=TRUE)`.

**Single-dataset digit-by-digit comparison**:
- ML: hand-coded EM vs. `lme4`(ML): $`\max|\Delta\gamma|=10^{-9}`$, $`\max|\Delta D|=2\times10^{-6}`$, log-likelihood difference $`2\times10^{-12}`$.
- REML: hand-coded EM vs. `lme4`(REML): $`\max|\Delta\gamma|=2\times10^{-9}`$, $`\max|\Delta D|=4\times10^{-6}`$, $`|\Delta\sigma^2|=3\times10^{-8}`$, log-likelihood difference $`4\times10^{-12}`$, confirming that **both** REML corrections — $`B_jCB_j^T`$ for $`D`$ (§4.5.2) and $`\mathrm{tr}(A_jCA_j^T)`$ for $`\sigma^2`$ (§4.5.3) — are derived correctly.

**200 simulations (bias/MSE evaluation, true $`D`$ diagonal $`(5,6,7)`$, $`\sigma^2=3`$)**:

| Method | Diagonal of $`\hat D`$ | $`\hat\sigma^2`$ | Mean max-diff vs. corresponding `lme4` |
|---|---|---:|---|
| Hand-coded EM(ML) | (4.076, 4.950, 5.718) | 2.986 | $`3\times10^{-6}`$ (vs. `lme4` ML) |
| Hand-coded EM(REML) | (5.104, 6.197, 7.158) | 2.986 | $`6\times10^{-6}`$ (vs. `lme4` REML) |

The ML diagonal of $`\hat D`$ is approximately $`(J-(q+1))/J=16/20=0.80`$ times the truth — observed values $`(4.08,4.95,5.72)\approx0.80\times(5,6,7)`$, exactly matching the bias magnitude in §4.5.2; REML correction recovers it to $`\approx102\%`$ of the truth. Under both estimands the hand-coded implementation agrees digit-for-digit with `lme4`, **confirming that the ML downward bias is an intrinsic property of the estimator, not an implementation defect**. (This 200-simulation validation uses a fixed $`D`$ with diagonal $`(5,6,7)`$ — a controlled setup for the `lme4` comparison — distinct from the QR-random true $`D`$ used in the §6.2 main simulation.)

### 6.3 General Linear Mixed Model Simulation Results

Parameter setup (seed=2025, 500 simulations):

```math
m=50,\quad n_i\in\{15,20,25\},\quad \beta=(1,2,-1,0.5),\quad
G_0=\begin{pmatrix}4&1\\1&2\end{pmatrix},\quad \sigma^2=1.
```

<p align="center">
 <img src="code/linear-mixed-model/figures/41_lmm_em_500sim_iter.png" width="45%" alt="General LMM: EM 500-simulation averaged iteration path">
 <img src="code/linear-mixed-model/figures/42_lmm_mcem_M200_500sim_iter.png" width="45%" alt="General LMM: MCEM(M=200) 500-simulation averaged iteration path">
</p>

> Figures show the **500-simulation averaged** iteration trajectory — averaging over independent datasets cancels single-dataset sampling noise, so the curves track the true values and the table means (a single-run trajectory would not). Generated by `code/linear-mixed-model/driver_compare500.R` (`set.seed(2025)`), saved in `code/linear-mixed-model/figures/`. The table reports EM and MCEM at Monte Carlo sample sizes $`M\in\{20,50,100,200\}`$, all averaged over 500 simulations.

| Metric | EM | MCEM $`M=20`$ | MCEM $`M=50`$ | MCEM $`M=100`$ | MCEM $`M=200`$ |
|---|---:|---:|---:|---:|---:|
| $`\hat\beta`$ | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) |
| Diagonal of $`\hat G_0`$ | (3.921, 1.953) | (3.920, 1.953) | (3.921, 1.953) | (3.921, 1.952) | (3.921, 1.952) |
| $`\hat\sigma^2`$ | 0.9950 | 0.9947 | 0.9951 | 0.9950 | 0.9950 |
| Total MSE | 0.13292 | 0.13329 | 0.13214 | 0.13369 | 0.13305 |
| Mean iterations | 8.34 | 50.00 (limit) | 50.00 (limit) | 50.00 (limit) | 50.00 (limit) |
| Total time (s) † | 30.6 | 441 | 824 | 1525 | 2839 |

> † All five configurations are run on the **same** 500 datasets (a paired comparison), so the only thing varying across the MCEM columns is the Monte Carlo sample size $`M`$. Wall times are from sequential execution (per-config compute is independent of the data).

Key findings:

- On the same datasets, EM and MCEM at every $`M`$ recover the true parameters and give essentially identical estimates: $`\hat\beta`$ is **identical** (it is the closed-form GLS solution, unaffected by the Monte Carlo E-step), and $`\hat G_0`$ / $`\hat\sigma^2`$ differ only by tiny Monte Carlo noise. MCEM reproduces EM's estimate even at $`M=20`$; larger $`M`$ merely smooths the iteration trajectory further (the $`M=200`$ path is the cleanest).
- The slight underestimation of the diagonal of $`\hat G_0`$ ($`\approx(3.92, 1.95)`$ vs. $`(4, 2)`$) is the intrinsic ML downward bias in variance components (which diminishes as the number of groups $`m`$ grows; REML in §6.3.1 removes it); the off-diagonal, noisy on a single dataset, recovers to $`\approx1`$ when averaged over 500 runs.
- MCEM hits the 50-iteration limit and its cost scales roughly linearly with $`M`$ — from $`\approx14\times`$ ($`M=20`$) to $`\approx93\times`$ ($`M=200`$) the EM wall time — with **no accuracy gain** here, because the E-step already has a closed form. MCEM pays off only when the E-step is intractable.

#### 6.3.1 Comparison with Library Results

`code/linear-mixed-model/driver_library_compare.R` (`set.seed(2025)`, $`m=50`$, 200 simulations) benchmarks the hand-coded EM/MCEM against two standard libraries, `lme4::lmer` and `nlme::lme`, under both ML and REML. Each row is aligned to the matching objective: hand-coded ML ↔ library ML, hand-coded REML ↔ library REML.

**Single-dataset digit-by-digit comparison**: the hand-coded EM agrees digit-for-digit with `lme4` and `nlme` under the matching criterion — ML differences are around $`10^{-7}`$ in variance components, and REML differences are the same order after adding the fixed-effect degrees-of-freedom corrections.

**Mean comparison over 200 simulations** (true values $`\beta=(1,2,-1,0.5)`$, $`G_0`$ diagonal $`(4,2)`$, $`\sigma^2=1`$):

| Method | Mean $`\hat\beta`$ | Diagonal of $`\hat G_0`$ | $`\hat\sigma^2`$ | Mean per-run MSE † |
|---|---|---|---:|---:|
| Hand-coded EM | (0.990, 1.989, −0.996, 0.504) | (3.987, 1.952) | 0.9976 | 1.2854 |
| `lme4`(ML) | (0.990, 1.989, −0.996, 0.504) | (3.987, 1.952) | 0.9976 | 1.2854 |

> † Analogous in spirit to §6.1.1, but the **convention differs**: here "mean per-run MSE" is the mean over simulations of the error *norm* $`\lVert\hat\beta-\beta\rVert_2+\lVert\hat G_0-G_0\rVert_F+|\hat\sigma^2-\sigma^2|`$ (the `get_mse` convention for variance-component models), whereas §6.1.1 sums *squared* errors — so the two tables' per-run columns are not numerically comparable. It measures single-run estimation accuracy; the "total MSE" (0.13292) in the main table of §6.3 is instead computed by averaging the 500 estimates first and then taking the deviation norm, measuring estimator bias — the two cannot be compared directly.

The hand-coded EM has a mean $`\max`$-diff of $`4.8\times10^{-7}`$ with `lme4`(ML) over 200 simulations and agrees equally digit-for-digit with `nlme::lme`, **confirming that the hand-coded EM is the standard LMM maximum-likelihood estimator**. If ML variance components are not satisfactory, the same driver now reports REML side-by-side; the REML path uses `options(lmm_reml=TRUE)` and matches `lme4(REML=TRUE)` / `nlme(method="REML")`. The $`\beta`$ in MCEM is computed by the closed-form GLS (so it is identical across $`M`$, as the §6.3 table shows); only the variance components carry Monte Carlo noise, which shrinks with $`M`$ while remaining close to the true values.

## 7. Empirical Analysis (Real Deep-Learning Data)

This corresponds to Chapter 5 of the thesis. Two pipelines **directly reuse** the EM/MCEM implementations from Chapters 3 and 4 on real data (only the data loading is replaced); complete reproduction instructions are in `code/empirical_README.md`. The random seed throughout is `20250529`.

### 7.1 Student's t Regression — UTKFace Facial Age

- **Data and task**: 8000 faces from Hugging Face `py97/UTKFace-Cropped`; frozen ResNet-50 embeddings (2048-dim) → PCA to 30 dimensions; regression target is age. $`n=8000`$, 31 predictors (including intercept).
- **Method**: The degrees of freedom $`\nu`$ is unknown; profile marginal likelihood over the grid $`\{1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50\}`$ selects $`\nu^*`$ (evaluated at the EM/MAP fit, whose variance differs from pure MLE only by the $`n/(n+2)`$ factor); EM and MCEM ($`M=20/50/200`$) are run at $`\nu^*`$; `optim` directly maximizing the marginal t likelihood and `MASS::rlm` provide two independent cross-checks. To support $`n=8000`$, the implementation uses row-scaled $`wX`$ instead of $`\mathrm{diag}(w)`$ (mathematically equivalent to the weighted least squares in §3.5).
- **Key results**: $`\nu^*=6`$; marginal t log-likelihood $`-32029.9`$, an **improvement of 151.0** over the normal baseline (OLS, $`-32180.9`$); maximum fixed-effect coefficient difference between EM(MAP) and `optim` marginal MLE is $`0.00019`$; correlation of EM and `rlm` coefficients $`\approx1.0`$; E-step weights $`\gamma_i`$ decrease monotonically with absolute residual — among the 10% most strongly down-weighted observations, ages $`\ge 60`$ account for $`59.5\%`$ (vs. $`11.3\%`$ in the full sample), meaning sparse elderly observations are automatically down-weighted.

| Coefficient | OLS | EM(t) | MCEM $`M=20`$ | MCEM $`M=200`$ | optim |
|---|---:|---:|---:|---:|---:|
| Intercept | 33.276 | 32.410 | 32.426 | 32.402 | 32.410 |
| PC1 | 2.383 | 2.444 | 2.468 | 2.445 | 2.444 |
| PC2 | 10.648 | 10.687 | 10.658 | 10.685 | 10.687 |

Figures generated by `code/empirical-t-utkface/code/03_replot.R`, saved in `code/empirical-t-utkface/output/`:

<p align="center">
 <img src="code/empirical-t-utkface/output/71_qq_normal.png" width="45%" alt="OLS residual normal QQ plot (tail deviations indicate heavy tails)">
 <img src="code/empirical-t-utkface/output/72_profile_nu.png" width="45%" alt="Profile marginal likelihood for degrees of freedom nu">
</p>
<p align="center">
 <img src="code/empirical-t-utkface/output/73_weights.png" width="45%" alt="E-step weights decrease monotonically with residual">
 <img src="code/empirical-t-utkface/output/74_qq_t.png" width="45%" alt="t(nu*) quantile-quantile plot (heavy tails absorbed by model)">
</p>

### 7.2 Two-Level Linear Model — CIFAR-10H Human Reaction Times

- **Data and task**: CIFAR-10H annotator reaction times for CIFAR-10 test images. Level-1: $`\log RT_{ij}=b_{0j}+b_{1j}\,\mathrm{diff}_{ij}+b_{2j}\,\mathrm{correct}_{ij}+b_{3j}\,\mathrm{trial}_{ij}+\varepsilon_{ij}`$ (covariates: difficulty, correctness, trial index; difficulty and trial are z-standardized to mean 0, SD 1 before fitting, so $`b_{1j}`$ and $`b_{3j}`$ are per-standard-deviation effects, while correctness is binary); level-2: $`\beta_j \sim N(\gamma, D)`$ (annotator random coefficients, $`q=0`$, $`W_j=I_4`$). After removing annotators with rank-deficient within-group design matrices (0% or 100% correct): $`J=299`$, $`N=59699`$.
- **Method**: **Direct reuse** of the Chapter 4 two-level model EM/MCEM (only the `get_X_j/y_j/W_j` accessors are overridden to read data by group, avoiding a massive block-diagonal matrix); EM and MCEM ($`M=20/50/200`$) are run and compared against `lme4` (ML) as gold standard.
- **Key results**: Fixed effects $`\gamma`$ = (intercept 7.648, difficulty 0.130, correct −0.257, trial −0.033); $`\sigma^2=0.1191`$ matches `lme4` exactly; marginal log-likelihood $`-22423.06`$ agrees with `lme4` **digit-for-digit** (difference $`\lt 0.1`$); random-effects SDs are $`(0.327, 0.050, 0.220, 0.038)`$; intercept ICC $`=0.473`$ (proportion of between-annotator baseline speed variation); MCEM (all $`M`$) agrees with EM to approximately 4 decimal places; EM marginal log-likelihood is monotonically increasing on real data.

| Fixed effect | EM | MCEM $`M=20`$ | MCEM $`M=200`$ | lme4 |
|---|---:|---:|---:|---:|
| Intercept | 7.6477 | 7.6475 | 7.6478 | 7.6477 |
| Difficulty | 0.1299 | 0.1299 | 0.1299 | 0.1299 |
| Correct | −0.2572 | −0.2569 | −0.2572 | −0.2572 |
| Trial | −0.0334 | −0.0334 | −0.0334 | −0.0334 |

Figures generated by `code/empirical-2level-cifar10h/code/03_replot.R`, saved in `code/empirical-2level-cifar10h/output/`:

<p align="center">
 <img src="code/empirical-2level-cifar10h/output/61_loglik_monotone.png" width="45%" alt="EM marginal log-likelihood monotonically increasing on real data">
 <img src="code/empirical-2level-cifar10h/output/62_caterpillar_intercept.png" width="45%" alt="Annotator random intercept caterpillar plot">
</p>
<p align="center">
 <img src="code/empirical-2level-cifar10h/output/63_gamma_convergence.png" width="45%" alt="EM convergence trajectory for fixed effects gamma">
</p>

### 7.3 Empirical Summary

The three patterns observed in simulation — monotonicity of the EM observed-data log-likelihood, the MCEM trade-off between sample size and accuracy/cost, and the consistency of EM and MCEM with sufficient sampling — all hold on the two real deep-learning datasets; and t regression EM (vs. `optim`/`rlm`) and two-level model EM (vs. `lme4`) each agree digit-for-digit with independent gold standards, validating the correctness of the implementations.

**Complete closed-loop library cross-validation.** This repository includes library cross-checks at both the **simulation** and **empirical** ends: simulation side in §6.1.1 (`optim`/`hett::tlm`/`MASS::rlm`), §6.2.1 (`lme4` under ML and REML), §6.3.1 (`lme4`/`nlme` under ML and REML), each reproduced by the corresponding model's `driver_library_compare.R`; empirical side in §7.1 (`optim`/`rlm`) and §7.2 (`lme4`). The conclusions at both ends are consistent: all three hand-coded EM/MCEM implementations agree with ecosystem gold standards to machine precision; any discrepancies arise solely from deliberate prior/estimand choices (the $`n+2`$ divisor in the t regression MAP, and the ML vs. REML variance components in mixed models), and each of those differences can be explained by "switching to the corresponding library's estimand and reproducing digit-for-digit."

## 8. Repository Structure

```text
.
├── README.md                     # This document: EM/MCEM derivations + model applications + empirical results
├── Algorithms.tex                  # LaTeX thesis source; Algorithms.pdf is a committed build artifact (regenerated via xelatex)
├── LICENSE                         # MIT License (Copyright (c) 2026 Li Xiuyin)
└── code/
    ├── README.md                 # Overview of all three model codebases and shared conventions
    ├── docs/                      # code_review_report.md — adversarial code review (27 findings / 21 confirmed)
    ├── empirical_README.md       # Reproduction instructions for the two empirical pipelines
    ├── general/                  # make_em_concept.R — EM lower-bound illustration (Fig. 2.1)
    ├── t-regression/             # Chapter 3: utils.R / em.R / mcem.R / driver_*.R / README.md
    ├── two-level-model/          # Chapter 4: same + figures/
    ├── linear-mixed-model/       # Chapter 4 extension: same + lmm_derivation.md
    ├── empirical-t-utkface/      # code/ data/ output/ (UTKFace t regression)
    └── empirical-2level-cifar10h/ # code/ data/ output/ (CIFAR-10H two-level model)
```

Code and documentation entry points:

| Path | Contents |
|---|---|
| `code/README.md` | Overview of all three model codebases, shared conventions, how to run, summary of findings |
| `code/t-regression/` | t regression EM/MCEM simulation (`README.md` contains iteration formulas and validation conclusions) |
| `code/two-level-model/` | Two-level linear model EM/MCEM simulation (`README.md`) |
| `code/linear-mixed-model/` | General LMM EM/MCEM simulation; `lmm_derivation.md` contains the complete derivation |
| `code/empirical-t-utkface/` | UTKFace deep-embedding t regression empirical analysis |
| `code/empirical-2level-cifar10h/` | CIFAR-10H reaction time two-level model empirical analysis |
| `code/docs/code_review_report.md` | Adversarial code review report for all three codebases |

## References

1. Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Wu, C. F. J. (1983). On the convergence properties of the EM algorithm.
3. Wei, G. C. G., and Tanner, M. A. (1990). A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms.
4. Liu, C., and Rubin, D. B. (1995). ML estimation of the t distribution using EM and its extensions.

## License

Released under the MIT License — see [LICENSE](LICENSE) (Copyright (c) 2026 Li Xiuyin).
