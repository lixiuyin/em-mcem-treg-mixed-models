# EM 与 MCEM 算法在一般混合线性模型中的应用（推导）

> [English](lmm_derivation.md) | **中文**

> 本文档给出一般混合线性模型（Linear Mixed Model, LMM）下 EM 与 MCEM 迭代公式的完整推导，
> 与论文第 4 章两层线性模型采用同一套贝叶斯/EM 框架与记号约定（平坦无信息先验、随机效应协方差 $G_0$ 的 M 步除以组数 $m$、$\sigma^2$ 除以总样本量 $n$）。
> 两层线性模型是本模型的一个特例。

## 1. 模型设定

对 $i=1,\dots,m$ 个二级单元（组），

$$
y_i = X_i\beta + Z_i u_i + \varepsilon_i,\qquad
u_i \sim \mathcal N(0, G_0),\quad \varepsilon_i \sim \mathcal N(0,\sigma^2 I_{n_i}),
$$

其中 $X_i\in\mathbb R^{n_i\times p}$ 为固定效应设计（含截距列），$Z_i\in\mathbb R^{n_i\times k}$ 为随机效应设计，
$u_i$（$k$ 维随机效应）与 $\varepsilon_i$ 相互独立，各组之间相互独立。
本实现取 $Z_i = X_i[\,,1{:}k]$，即“随机截距 + 前 $k-1$ 个协变量的随机斜率”（随机系数模型）。

记 $n=\sum_i n_i$，堆叠形式为 $y=X\beta+Zu+\varepsilon$，$u\sim\mathcal N(0, I_m\otimes G_0)$，$\varepsilon\sim\mathcal N(0,\sigma^2 I_n)$。
待估参数为固定效应 $\beta$、随机效应协方差 $G_0$（$k\times k$）、个体误差方差 $\sigma^2$。

**与两层线性模型的关系**：取 $X_i=X_j W_j$、$Z_i=X_j$、$u_i=\mu_j$、$\beta=\gamma$、$G_0=D$，即得论文第 4 章的两层模型。故本模型是其推广（固定与随机设计可彼此不同、任意指定）。

## 2. 相关分布

**(1) 观测数据边际分布。** 由 $y_i = X_i\beta + Z_i u_i + \varepsilon_i$ 及独立性，

$$
y_i \sim \mathcal N\!\big(X_i\beta,\; V_i\big),\qquad V_i = Z_i G_0 Z_i^{T} + \sigma^2 I_{n_i}.
$$

**(2) 隐变量后验。** 将 $u$ 视为隐变量，$(y,u)$ 为完全数据。各组相互独立，故

$$
u_i \mid y_i \sim \mathcal N(u_i^{*}, V_i^{*}),\qquad
u_i^{*}=G_0 Z_i^{T}V_i^{-1}(y_i-X_i\beta),\quad
V_i^{*}=G_0 - G_0 Z_i^{T}V_i^{-1}Z_i G_0 .
$$

由 Woodbury 矩阵恒等式 $V_i^{-1}=\sigma^{-2}I-\sigma^{-2}Z_i(\sigma^2G_0^{-1}+Z_i^{T}Z_i)^{-1}Z_i^{T}$，上式可化为便于计算的形式：

$$
\boxed{\,u_i^{*}=\big(Z_i^{T}Z_i+\sigma^2 G_0^{-1}\big)^{-1}Z_i^{T}(y_i-X_i\beta),\qquad
V_i^{*}=\sigma^2\big(Z_i^{T}Z_i+\sigma^2 G_0^{-1}\big)^{-1}.}
$$

**(3) 完全数据条件分布。** $y_i\mid u_i \sim \mathcal N\!\big(X_i\beta+Z_i u_i,\ \sigma^2 I_{n_i}\big)$。

## 3. 先验与完全数据后验对数似然

取平坦无信息先验 $\pi(\beta,G_0,\sigma^2)\propto 1$。在此先验下最大后验估计与最大似然估计一致，故 EM 实际极大化的是观测数据边际对数似然 $\ln p(y\mid\beta,G_0,\sigma^2)$。

完全数据后验对数似然（仅保留与参数有关的部分）：

$$
\ln L(\beta,G_0,\sigma^2\mid y,u)\propto
-\frac{n}{2}\ln\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\lVert y_i-X_i\beta-Z_i u_i\rVert^2
-\frac{m}{2}\ln\lvert G_0\rvert
-\frac{1}{2}\sum_{i=1}^{m}u_i^{T}G_0^{-1}u_i .
$$

## 4. EM 算法

**E 步（第 $t$ 次迭代）。** 用 $\hat\beta_t,\hat G_{0,t},\hat\sigma^2_t$ 计算后验 $u_i\mid y_i\sim\mathcal N(u_{i,t}^{*},V_{i,t}^{*})$，得 $Q$ 函数（用到
$\mathbb E[u_i\mid y]=u_i^{*}$，$\mathbb E[u_iu_i^{T}\mid y]=u_i^{*}u_i^{*T}+V_i^{*}$）：

$$
\begin{aligned}
Q\propto&-\frac{n}{2}\ln\sigma^2
-\frac{1}{2\sigma^2}\sum_{i}\Big[\lVert y_i-X_i\beta-Z_i u_i^{*}\rVert^2+\mathrm{tr}(Z_i V_i^{*}Z_i^{T})\Big]\\
&-\frac{m}{2}\ln\lvert G_0\rvert-\frac{1}{2}\sum_i\Big[u_i^{*T}G_0^{-1}u_i^{*}+\mathrm{tr}(G_0^{-1}V_i^{*})\Big].
\end{aligned}
$$

**M 步。**

- **$\beta$ 更新。** 严格 EM 对 $\beta$ 求偏导得 $\hat\beta=\big(\sum_i X_i^{T}X_i\big)^{-1}\sum_i X_i^{T}(y_i-Z_i u_i^{*})$。本实现采用等价但收敛更快的**闭式 GLS**（给定 $G_0,\sigma^2$ 时 $\beta$ 的边际极大似然，即 ECME 步）：

$$
\boxed{\;\hat\beta_{t+1}=\Big(\sum_{i=1}^{m}X_i^{T}V_i^{-1}X_i\Big)^{-1}\sum_{i=1}^{m}X_i^{T}V_i^{-1}y_i,\qquad V_i=Z_iG_0Z_i^{T}+\sigma^2 I_{n_i}.\;}
$$

二者收敛点相同（均为边际极大似然），GLS 直接最大化边际似然、避免方差分量较大时的缓慢收敛。

- **$G_0$ 更新。** 对 $G_0$ 求偏导（利用 $\partial\ln|G_0|/\partial G_0=G_0^{-1}$、$\partial(u^{T}G_0^{-1}u)/\partial G_0=-G_0^{-1}uu^{T}G_0^{-1}$）并令其为 0：

$$
m\,G_0=\sum_i\big(u_i^{*}u_i^{*T}+V_i^{*}\big)\ \Longrightarrow\
\boxed{\;\hat G_{0,t+1}=\frac{1}{m}\sum_{i=1}^{m}\big(u_i^{*}u_i^{*T}+V_i^{*}\big).\;}
$$

- **$\sigma^2$ 更新。**

$$
\boxed{\;\hat\sigma^2_{t+1}=\frac{1}{n}\sum_{i=1}^{m}\Big[\lVert y_i-X_i\hat\beta_{t+1}-Z_i u_i^{*}\rVert^2+\mathrm{tr}(Z_i V_i^{*}Z_i^{T})\Big].\;}
$$

### 4.1 方差分量的 REML 校正

上面的 ML 公式使用的是正确的 ML 除数（$G_0$ 除以 $m$，$\sigma^2$ 除以 $n$），但它们把 GLS 估计 $\hat\beta$ 当作已知量。组数有限时，估计固定效应会消耗自由度，使方差分量偏小。REML 的做法是把 $\hat\beta$ 的不确定性传播回二阶矩。

按 ECM 次序（见下），两个 M 步在**不同**迭代处评价各自的 REML 量——$G_0$ 更新在更新前的 $G_{0,t}$ 处，$\sigma^2$ 更新在刚更新的 $G_{0,t+1}$ 处——仅在不动点处重合。记 $V_{i,s}=Z_iG_{0,s}Z_i^T+\hat\sigma_t^2 I$，

$$
C_s=\left(\sum_i X_i^T V_{i,s}^{-1}X_i\right)^{-1}.
$$

对 $G_0$（量在 $G_{0,t}$ 处），令 $B_{i,t}=G_{0,t}Z_i^TV_{i,t}^{-1}X_i$，REML 更新为

$$
\boxed{\;G_{0,t+1}^{\mathrm{REML}}=\frac1m\sum_i\left(u_i^*u_i^{*T}+V_i^*+B_{i,t}C_tB_{i,t}^T\right).\;}
$$

对 $\sigma^2$（量在 $G_{0,t+1}$ 处），令 $A_{i,t+1}=X_i-Z_iG_{0,t+1}Z_i^TV_{i,t+1}^{-1}X_i$，REML 更新为

$$
\boxed{\;\sigma_{t+1}^{2,\mathrm{REML}}=\frac1n\sum_i\left[
\lVert y_i-X_i\hat\beta_{t+1}-Z_i u_i^*\rVert^2+\mathrm{tr}(Z_iV_i^*Z_i^T)+\mathrm{tr}(A_{i,t+1}C_{t+1}A_{i,t+1}^T)\right].\;}
$$

这与代码一致：`get_reml_pieces` 在 `em_update_G0` 中以更新前的 `G0` 调用，在 `em_update_sigma2` 中以更新后的 `G0_new` 调用。

这里除数仍为 $n$；迹项补回固定效应自由度。若没有随机效应，该固定点退化为通常的残差方差 $RSS/(n-p)$。在 `options(lmm_reml=TRUE)` 下，`get_loglik` 返回限制对数似然

$$
\ell_R=\ell_{\mathrm{ML}}(\hat\beta,G_0,\sigma^2)-\frac12\log\left|\sum_i X_i^TV_i^{-1}X_i\right|+\frac{p}{2}\log(2\pi).
$$

实现采用 ECM 次序：先更新 $\beta$（闭式 GLS，不依赖后验），再用最新的 $(\hat\beta_{t+1},\hat G_{0,t},\hat\sigma^2_t)$ 计算后验更新 $G_0$，再用 $(\hat\beta_{t+1},\hat G_{0,t+1},\hat\sigma^2_t)$ 的后验更新 $\sigma^2$。每步均不减相应的单调目标：ML 口径下为观测数据边际对数似然，REML 口径下为限制对数似然 $\ell_R$（由 `get_loglik` 返回）。

## 5. MCEM 算法

当后验无法解析积分时（本模型仍可解析，此处用作演示与对照），E 步改用蒙特卡洛近似：从 $u_i\mid y_i\sim\mathcal N(u_i^{*},V_i^{*})$ 抽取 $M$ 个样本 $u_i^{(1)},\dots,u_i^{(M)}$，则

$$
\hat G_{0,t+1}=\frac{1}{mM}\sum_{l=1}^{M}\sum_{i=1}^{m}u_i^{(l)}u_i^{(l)T},\qquad
\hat\sigma^2_{t+1}=\frac{1}{nM}\sum_{l=1}^{M}\sum_{i=1}^{m}\lVert y_i-X_i\hat\beta_{t+1}-Z_i u_i^{(l)}\rVert^2 .
$$

$\beta$ 仍用闭式 GLS（解析可算，无需采样）。当 $M\to\infty$，MCEM 各更新依大数定律收敛到对应 EM 更新。

在 `options(lmm_reml=TRUE)` 下，MCEM 的 E 步与 §4.1 的 EM REML 校正一致：$G_0$ 的样本从膨胀协方差 $\mathcal N\!\big(u_i^{*},\,V_i^{*}+B_{i,t}C_tB_{i,t}^T\big)$ 抽取，使样本二阶矩复现 REML 的 $G_0$ 更新；而 $\sigma^2$ 的样本仍用 ML 协方差 $\mathcal N(u_i^{*},V_i^{*})$，并显式加入校正项 $\sum_i\mathrm{tr}(A_{i,t+1}C_{t+1}A_{i,t+1}^T)$（避免重复计数）——与 `get_u_samples`、`mcem_update_sigma2` 对应。

## 6. 单调性与收敛

平坦先验下 ML 口径的 EM/ECME 极大化目标为观测数据边际对数似然
$\ln p(y\mid\beta,G_0,\sigma^2)=\sum_i\big[-\tfrac{n_i}{2}\ln 2\pi-\tfrac12\ln|V_i|-\tfrac12(y_i-X_i\beta)^{T}V_i^{-1}(y_i-X_i\beta)\big]$，
该量在每步迭代单调不减（ML 口径下代码 `get_loglik` 即计算此量；REML 口径下 `get_loglik` 返回 §4.1 的限制对数似然 $\ell_R$，此时单调目标即为 $\ell_R$），并已在 $m=50$ 模拟中验证单调。
方差分量的极大似然估计存在已知的向下偏（估计固定效应消耗自由度），故 $\hat G_0$、$\hat\sigma^2$ 略偏小。REML 使用限制对数似然与 §4.1 的校正项来消除此自由度偏差；代码通过 `options(lmm_reml=TRUE)` 切换。

## 7. 实现对应

| 公式 | 代码（`utils.R`） |
|---|---|
| 后验 $u_i^{*},V_i^{*}$ | `get_post_u` |
| $V_i$ | `get_V_i` |
| $\hat\beta$（GLS） | `em_update_beta` |
| $\hat G_0$（ML ÷$m$；REML 加 $B_iCB_i^T$） | `em_update_G0` |
| $\hat\sigma^2$（ML ÷$n$；REML 加 $\mathrm{tr}(A_iCA_i^T)$） | `em_update_sigma2` |
| MCEM $\hat G_0,\hat\sigma^2$ | `mcem_update_G0`, `mcem_update_sigma2`, `mcem_update_all` |
| 边际 / 限制对数似然 | `get_loglik` |
