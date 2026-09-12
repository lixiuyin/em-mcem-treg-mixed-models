# EM 与 MCEM 算法：完整推导、模型应用与实证结果

> **简体中文** | [English](README.en.md)

本仓库整理本科毕业论文《EM 算法、MCEM 算法的原理及其在回归模型参数估计中的应用》的核心内容，给出 EM、MCEM、Student's t 回归和两层线性模型的完整推导过程，并附实现代码与实证结果。

为保证 GitHub 渲染稳定，独立公式用 `math` 代码块，行内公式统一用 `` $`...`$ `` 形式（反引号包裹，避免与中文标点相邻时 `$...$` 不渲染）。本文采用修订稿中的最终约定：

- t 回归中取先验 $`\pi(\beta, \sigma^2)\propto 1/\sigma^2`$，因此 $`\sigma^2`$ 的 MAP 更新除数为 $`n+2`$。

- 两层线性模型中取平坦无信息先验 $`\pi(\gamma, D, \sigma^2)\propto 1`$，因此 $`D`$ 的 EM 更新除数为 $`J`$，MCEM 更新除数为 $`MJ`$。

- 固定效应（$`\beta`$ / $`\gamma`$）一律采用边际 GLS 闭式更新（即 ECME 步）；Monte Carlo 近似只作用于方差分量（$`D`$、$`\sigma^2`$）。
- 各模型实现均按 **multicycle-ECM** 次序：每更新完一个参数块，就用最新参数刷新 E 步（重算条件期望 / 重新抽样）再更新下一块。该次序与一次性 E 步的标准 EM 共享同一组不动点，且观测数据边际对数似然仍逐步单调不减（详见 §4.5、§4.6）。
- 一般混合线性模型（§5）是两层模型的推广，沿用同一套约定：平坦先验，ML 口径下 $`G_0`$ 的 EM 除数为组数 $`m`$（MCEM 为 $`Mm`$），$`\sigma^2`$ 除以总样本量 $`n`$（MCEM 为 $`Mn`$）。若 ML 方差分量明显偏小，可用 `options(lmm_reml=TRUE)` 启用 REML 并并列报告。

## 目录

- [1. 一般 EM 算法](#1-一般-em-算法)
- [2. 一般 MCEM 算法](#2-一般-mcem-算法)
- [3. Student's t 回归模型](#3-students-t-回归模型)
- [4. 两层线性模型](#4-两层线性模型)
- [5. 一般混合线性模型](#5-一般混合线性模型)
- [6. 数值模拟结论](#6-数值模拟结论)
- [7. 实证分析（深度学习真实数据）](#7-实证分析深度学习真实数据)
- [8. 仓库结构](#8-仓库结构)

## 1. 一般 EM 算法

设 $`Y`$ 为观测数据，$`Z`$ 为隐变量或缺失数据，参数为 $`\theta`$。观测数据似然为

```math
L(\theta\mid Y)=p(Y\mid\theta)=\int p(Y,Z\mid\theta)\,dZ.
```

直接最大化该积分通常困难。EM 的思想是引入当前参数 $`\theta_k`$ 下的条件分布 $`p(Z\mid Y,\theta_k)`$，并构造完全数据对数似然的条件期望：

```math
Q(\theta\mid\theta_k)
=E_{\theta_k}[\log p(Y,Z\mid\theta)\mid Y].
```

### 1.1 Jensen 不等式推导

对任意密度 $`q(Z)`$ 有

```math
\log p(Y\mid\theta)
=\log\int q(Z)\frac{p(Y,Z\mid\theta)}{q(Z)}\,dZ.
```

由 Jensen 不等式，

```math
\log p(Y\mid\theta)
\geq
\int q(Z)\log\frac{p(Y,Z\mid\theta)}{q(Z)}\,dZ.
```

取 $`q(Z)=p(Z\mid Y,\theta_k)`$，得到观测对数似然的下界

```math
B(\theta\mid\theta_k)
=Q(\theta\mid\theta_k)
-E_{\theta_k}[\log p(Z\mid Y,\theta_k)\mid Y].
```

其中第二项与待优化的 $`\theta`$ 无关，所以最大化 $`B`$ 等价于最大化 $`Q`$。

<p align="center">
 <img src="code/general/figures/em_concept.png" width="60%" alt="EM 下界示意图">
</p>

### 1.2 EM 迭代步骤

E 步：

```math
Q(\theta\mid\theta_k)
=E_{\theta_k}[\log p(Y,Z\mid\theta)\mid Y].
```

M 步：

```math
\theta_{k+1}
=\arg\max_{\theta}Q(\theta\mid\theta_k).
```

若 M 步只要求 $`Q(\theta_{k+1}\mid\theta_k)\ge Q(\theta_k\mid\theta_k)`$，则得到广义 EM（GEM）。

### 1.3 单调性

观测对数似然可分解为

```math
\log p(Y\mid\theta)
=Q(\theta\mid\theta_k)
-E_{\theta_k}[\log p(Z\mid Y,\theta)\mid Y].
```

进一步有

```math
\log p(Y\mid\theta)-\log p(Y\mid\theta_k)
=Q(\theta\mid\theta_k)-Q(\theta_k\mid\theta_k)
+KL(p(Z\mid Y,\theta_k)\,\|\,p(Z\mid Y,\theta)).
```

由于 KL 散度非负，若 M 步使 $`Q`$ 不下降，则

```math
\log p(Y\mid\theta_{k+1})
\ge
\log p(Y\mid\theta_k).
```

因此 EM 的观测数据对数似然单调不下降。其收敛点通常是似然函数的稳定点；若目标函数单峰，则可得到唯一极大似然估计。

## 2. 一般 MCEM 算法

当 $`Q(\theta \mid \theta_k)`$ 中的条件期望无法解析计算时，MCEM 用 Monte Carlo 样本近似 E 步。

从当前条件分布抽样：

```math
Z^{(1)},\dots,Z^{(M)}
\sim p(Z\mid Y,\theta_k).
```

用样本平均近似 $`Q`$ 函数：

```math
Q_M(\theta\mid\theta_k)
=\frac{1}{M}\sum_{m=1}^{M}\log p(Y,Z^{(m)}\mid\theta).
```

M 步改为

```math
\theta_{k+1}
=\arg\max_{\theta}Q_M(\theta\mid\theta_k).
```

当 $`M`$ 足够大时，由大数定律，$`Q_M(\theta\mid\theta_k)\to Q(\theta\mid\theta_k)`$，因此 MCEM 逼近 EM。代价是采样误差会使迭代路径产生随机波动，观测似然不再保证每一步严格单调；通常需随迭代增加 $`M`$ 或在固定 $`M`$ 下接受近似收敛。

## 3. Student's t 回归模型

### 3.1 模型与隐变量表示

设

```math
y_i=x_i^T\beta+\varepsilon_i,\qquad
\varepsilon_i\sim t_{\nu}(0,\sigma^2),\qquad i=1,\dots,n.
```

其中 $`x_i=(1,x_{i1},\dots,x_{ip})^T`$，$`\beta`$ 为 $`p+1`$ 维回归系数，自由度 $`\nu`$ 视为给定。写成矩阵形式，将 $`n`$ 个观测堆叠为 $`y=(y_1,\dots,y_n)^T`$、$`X=(x_1,\dots,x_n)^T\in\mathbb{R}^{n\times(p+1)}`$，

```math
y=X\beta+\varepsilon,\qquad \varepsilon\sim t_\nu(0,\sigma^2 I_n)\ \text{（各行独立）}.
```

t 分布可表示为正态-伽马混合。采用 Gamma 的 rate 参数化：

```math
z_i\sim \mathrm{Gamma}(\frac{\nu}{2},\frac{\nu}{2}),
```

```math
y_i\mid z_i,\beta,\sigma^2
\sim N(x_i^T\beta,\frac{\sigma^2}{z_i}).
```

因此 $`z_i`$ 可作为隐变量。给定 $`z_i`$ 后，模型变为加权正态回归。将隐尺度收进对角权重矩阵 $`W_z=\mathrm{diag}(z_1,\dots,z_n)`$，条件模型即异方差（精度加权）高斯，

```math
y\mid z,\beta,\sigma^2\sim N\!\big(X\beta,\ \sigma^2 W_z^{-1}\big),
```

即每个 $`z_i`$ 缩放第 $`i`$ 行的精度。这正是 EM 背后的矩阵图景：用 E 步期望 $`\gamma_i^{(k)}=E[z_i\mid y_i]`$（§3.3）替换未观测的 $`z_i`$，M 步即化为以权重矩阵 $`W_k=\mathrm{diag}(\gamma_1^{(k)},\dots,\gamma_n^{(k)})`$ 的加权最小二乘（§3.5）；残差越大则 $`z_i\downarrow`$，自动下调该观测的权重——这正是 t 回归稳健性的来源。

该表示的正确性可由边际化验证：对 $`z_i`$ 积分，

```math
p(y_i\mid\beta,\sigma^2)
=\int_0^\infty
N\!\left(y_i;x_i^T\beta,\frac{\sigma^2}{z_i}\right)
\mathrm{Gamma}\!\left(z_i;\frac{\nu}{2},\frac{\nu}{2}\right)dz_i,
```

被积函数关于 $`z_i`$ 是 $`\mathrm{Gamma}\!\left(\frac{\nu+1}{2}, \frac{1}{2}\left[\nu+\frac{(y_i-x_i^T\beta)^2}{\sigma^2}\right]\right)`$ 的核，积分给出归一化常数后正是自由度 $`\nu`$、尺度 $`\sigma^2`$ 的 Student's t 密度，故边际确为 $`t_\nu(x_i^T\beta,\sigma^2)`$。

### 3.2 完全数据后验对数似然

在该混合表示下，

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

所以完全数据密度满足

```math
p(y_i,z_i\mid\beta,\sigma^2)
=p(y_i\mid z_i,\beta,\sigma^2)p(z_i).
```

展开得

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

对 $`i=1,\dots,n`$ 求乘积，并只保留与 $`\beta, \sigma^2`$ 有关的部分：

```math
\log p(y,z\mid\beta,\sigma^2)
\propto
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}z_i(y_i-x_i^T\beta)^2.
```

论文采用先验 $`\pi(\beta,\sigma^2)\propto\frac{1}{\sigma^2}`$，故完全数据后验对数似然为

```math
\log \pi(\beta,\sigma^2\mid y,z)
\propto
-(\frac{n}{2}+1)\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}z_i(y_i-x_i^T\beta)^2.
```

### 3.3 条件分布 $`z_i\mid y_i`$

由 Bayes 公式，

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

它正是 Gamma 分布核。采用 rate 参数化，即若

```math
Z\sim \mathrm{Gamma}(a,b),
\qquad
p(z)\propto z^{a-1}\exp(-bz),
```

则此处

```math
z_i\mid y_i,\beta,\sigma^2
\sim
\mathrm{Gamma}(
\frac{\nu+1}{2},
\frac{1}{2}[\nu+\frac{(y_i-x_i^T\beta)^2}{\sigma^2}]
).
```

其条件期望为

```math
E[z_i\mid y_i,\beta,\sigma^2]
=
\frac{\nu+1}{\nu+(y_i-x_i^T\beta)^2/\sigma^2}.
```

在第 $`k`$ 次迭代，记

```math
\gamma_i^{(k)}
=
\frac{\nu+1}{\nu+(y_i-x_i^T\beta_k)^2/\sigma_k^2}.
```

### 3.4 t 回归 EM 的 E 步

代入条件期望，得到

```math
Q(\beta,\sigma^2\mid\beta_k,\sigma_k^2)
\propto
-(\frac{n}{2}+1)\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}
\gamma_i^{(k)}(y_i-x_i^T\beta)^2.
```

令 $`W_k=\mathrm{diag}(\gamma_1^{(k)},\dots,\gamma_n^{(k)})`$。

### 3.5 t 回归 EM 的 M 步

对 $`\beta`$ 求偏导：

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

令偏导为零：

```math
\sum_{i=1}^{n}\gamma_i^{(k)}x_i y_i
=
\sum_{i=1}^{n}\gamma_i^{(k)}x_i x_i^T\beta.
```

因此

```math
\beta_{k+1}
=
(X^T W_k X)^{-1}X^T W_k y.
```

对 $`\sigma^2`$ 求偏导：

```math
\frac{\partial Q}{\partial\sigma^2}
=
-\frac{n/2+1}{\sigma^2}
+\frac{1}{2(\sigma^2)^2}
\sum_{i=1}^{n}\gamma_i^{(k)}(y_i-x_i^T\beta)^2.
```

令偏导为零：

```math
-(\frac{n}{2}+1)\sigma^2
+\frac{1}{2}
\sum_{i=1}^{n}\gamma_i^{(k)}(y_i-x_i^T\beta)^2
=0.
```

整理得

```math
(n+2)\sigma^2
=
\sum_{i=1}^{n}\gamma_i^{(k)}(y_i-x_i^T\beta)^2.
```

代入 $`\beta_{k+1}`$：

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}
(y-X\beta_{k+1})^TW_k(y-X\beta_{k+1}).
```

这表明 t 回归 EM 等价于反复进行加权最小二乘。残差越大的样本，其权重越小，因此该模型具有稳健性。

### 3.6 t 回归 EM 算法流程

给定初始值 $`\beta_0, \sigma_0^2`$、自由度 $`\nu`$、最大迭代次数 $`K`$ 和收敛阈值 $`\varepsilon`$：

1. 对 $`k=0,1,2,\dots`$ 重复执行 E 步与 M 步。
2. E 步计算每个样本的条件权重：

```math
\gamma_i^{(k)}
=
\frac{\nu+1}{\nu+(y_i-x_i^T\beta_k)^2/\sigma_k^2},
\qquad i=1,\dots,n.
```

3. 令 $`W_k=\mathrm{diag}(\gamma_1^{(k)},\dots,\gamma_n^{(k)})`$。

4. M 步更新

```math
\beta_{k+1}=(X^TW_kX)^{-1}X^TW_ky.
```

5. 再更新

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}(y-X\beta_{k+1})^TW_k(y-X\beta_{k+1}).
```

6. 若 $`\max\!\big(\|\beta_{k+1}-\beta_k\|,\ \left|\sigma_{k+1}^2-\sigma_k^2\right|\big)\lt\varepsilon`$，则停止；否则令 $`k\leftarrow k+1`$ 继续迭代。

### 3.7 t 回归 MCEM

在 MCEM 中，不直接使用 $`E[z_i\mid y_i]`$，而是从条件分布抽样：

```math
z_i^{(k+1,m)}
\sim
\mathrm{Gamma}(
\frac{\nu+1}{2},
\frac{1}{2}[\nu+\frac{(y_i-x_i^T\beta_k)^2}{\sigma_k^2}]
),
\qquad m=1,\dots,M.
```

用样本均值近似条件期望：

```math
\bar z_i^{(k+1)}
=
\frac{1}{M}\sum_{m=1}^{M}z_i^{(k+1,m)}.
```

令 $`\bar W_k=\mathrm{diag}(\bar z_1^{(k+1)},\dots,\bar z_n^{(k+1)})`$。近似 Q 函数为

```math
Q_M(\beta,\sigma^2\mid\beta_k,\sigma_k^2)
\propto
-(\frac{n}{2}+1)\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{i=1}^{n}\bar z_i^{(k+1)}(y_i-x_i^T\beta)^2.
```

M 步与 EM 完全同型：

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

### 3.8 t 回归 MCEM 算法流程

给定初始值 $`\beta_0, \sigma_0^2`$、自由度 $`\nu`$、Monte Carlo 样本量 $`M`$、最大迭代次数 $`K`$ 和阈值 $`\varepsilon`$：

1. 对每次迭代 $`k`$，先计算当前条件分布参数

```math
a_i^{(k)}=\frac{\nu+1}{2},
\qquad
b_i^{(k)}
=
\frac{1}{2}\left[\nu+\frac{(y_i-x_i^T\beta_k)^2}{\sigma_k^2}\right].
```

2. 对每个 $`i`$ 抽取

```math
z_i^{(k+1,1)},\dots,z_i^{(k+1,M)}
\sim
\mathrm{Gamma}(a_i^{(k)},b_i^{(k)}).
```

3. 用样本均值近似 E 步期望：

```math
\bar z_i^{(k+1)}
=
\frac{1}{M}\sum_{m=1}^{M}z_i^{(k+1,m)}.
```

4. 构造 $`\bar W_k=\mathrm{diag}(\bar z_1^{(k+1)},\dots,\bar z_n^{(k+1)})`$。

5. 执行与 EM 同型的 M 步：

```math
\beta_{k+1}=(X^T\bar W_kX)^{-1}X^T\bar W_ky,
```

```math
\sigma_{k+1}^2
=
\frac{1}{n+2}(y-X\beta_{k+1})^T\bar W_k(y-X\beta_{k+1}).
```

6. 按参数变化量或最大迭代次数停止。由于采样误差存在，MCEM 的路径可能不严格单调。

## 4. 两层线性模型

### 4.1 模型设定

设第 $`j`$ 组有 $`n_j`$ 个观测：

```math
y_j=X_j\beta_j+\varepsilon_j,\qquad
\varepsilon_j\sim N(0,\sigma^2I_{n_j}),
```

组内回归系数满足第二层模型：

```math
\beta_j=W_j\gamma+\mu_j,\qquad
\mu_j\sim N(0,D),
```

其中 $`\gamma`$ 为固定效应，$`\mu_j`$ 为随机效应，$`D`$ 为随机效应协方差矩阵。代入后：

```math
y_j=X_jW_j\gamma+X_j\mu_j+\varepsilon_j.
```

记

```math
N=\sum_{j=1}^{J}n_j,\qquad
T=I_J\otimes D.
```

以及块对角设计与堆叠的二级结构——记 $`X_j\in\mathbb{R}^{n_j\times(p+1)}`$（$`p`$ 个组内协变量加截距）、组级设计 $`W_j\in\mathbb{R}^{(p+1)\times(p+1)(q+1)}`$（$`p+1`$ 个系数各对 $`q+1`$ 个组级协变量回归，故 $`\gamma\in\mathbb{R}^{(p+1)(q+1)}`$），

```math
X=\mathrm{diag}(X_1,\dots,X_J)\in\mathbb{R}^{N\times J(p+1)},\quad
W=\begin{pmatrix}W_1\\\vdots\\W_J\end{pmatrix}\in\mathbb{R}^{J(p+1)\times(p+1)(q+1)},\quad
\mu=\begin{pmatrix}\mu_1\\\vdots\\\mu_J\end{pmatrix}.
```

将所有组堆叠，有

```math
y=XW\gamma+X\mu+\varepsilon,\qquad
\mu\sim N(0,T),\qquad
\varepsilon\sim N(0,\sigma^2I_N).
```

因此观测数据边际分布为

```math
y\sim N(XW\gamma,V),\qquad
V=XTX^T+\sigma^2I_N.
```

由于 $`X`$ 与 $`T`$ 均为块对角，故 $`V=\mathrm{diag}(V_1,\dots,V_J)`$ 亦为块对角，其中 $`V_j=X_jDX_j^T+\sigma^2 I_{n_j}`$：各组相互独立，故每步 EM 更新都按组分解。

完全数据联合分布的协方差矩阵为修订稿中的正确形式：

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

注意右上块是 $`XT`$，左下块是 $`TX^T`$。

### 4.2 随机效应条件分布

由多元正态条件分布公式：

```math
\mu\mid y,\gamma,D,\sigma^2
\sim N(\mu^*,V^*),
```

其中

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

利用 Woodbury 恒等式可写为

```math
\mu^*
=(X^TX+\sigma^2T^{-1})^{-1}X^T(y-XW\gamma),
```

```math
V^*
=
\sigma^2(X^TX+\sigma^2T^{-1})^{-1}.
```

等价性可由"推移恒等式"验证：

```math
(X^TX+\sigma^2T^{-1})^{-1}X^T
=
TX^T(XTX^T+\sigma^2I_N)^{-1},
```

两边分别左乘 $`(X^TX+\sigma^2T^{-1})`$、右乘 $`(XTX^T+\sigma^2I_N)`$ 后都化为 $`X^T(XTX^T+\sigma^2I_N)`$，故恒等式成立，由此得 $`\mu^*`$ 的第二种写法。$`V^*`$ 则是 Woodbury 恒等式

```math
(T^{-1}+\sigma^{-2}X^TX)^{-1}
=
T-TX^T(XTX^T+\sigma^2I_N)^{-1}XT
```

的右端，且 $`(T^{-1}+\sigma^{-2}X^TX)^{-1}=\sigma^2(X^TX+\sigma^2T^{-1})^{-1}`$。由于各组独立，第 $`j`$ 组条件分布可写为

```math
\mu_j\mid y_j,\gamma,D,\sigma^2
\sim N(m_j,V_j^*),
```

其中

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

### 4.3 完全数据对数似然

采用平坦先验 $`\pi(\gamma,D,\sigma^2)\propto 1`$。此时 MAP 与 MLE 一致，EM 极大化的是观测数据边际对数似然 $`\log p(y \mid \gamma, D, \sigma^2)`$。完全数据对数似然中与参数有关的部分为

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

这里 $`\log|D|`$ 的系数是 $`-J/2`$，对应修订稿中的 $`D`$ 更新除数 $`J`$。

### 4.4 两层模型 EM 的 E 步

在第 $`k`$ 次迭代，使用当前估计 $`\gamma_k, D_k, \sigma_k^2`$ 计算

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

需要用到两个条件期望：

```math
E[\mu_j\mid y,\theta_k]=m_{jk},
```

```math
E[\mu_j\mu_j^T\mid y,\theta_k]
=m_{jk}m_{jk}^T+V_{jk}.
```

同时

```math
E[
\| y_j-X_jW_j\gamma-X_j\mu_j\|^2
\mid y,\theta_k
]
=
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T).
```

因此

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

### 4.5 两层模型 EM 的 M 步

#### 4.5.1 固定效应 $`\gamma`$

保留与 $`\gamma`$ 有关的项：

```math
\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2.
```

对 $`\gamma`$ 求偏导并令其为零：

```math
\sum_{j=1}^{J}
W_j^TX_j^T(y_j-X_jW_j\gamma-X_jm_{jk})
=0.
```

得到严格 EM 更新：

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^TX_j^TX_jW_j)^{-1}
\sum_{j=1}^{J}W_j^TX_j^T(y_j-X_jm_{jk}).
```

修订稿与代码采用边际 GLS 闭式更新作为 $`\gamma`$ 的实际更新方式。令

```math
\hat\beta_j^{OLS}
=(X_j^TX_j)^{-1}X_j^Ty_j,
```

则在边际模型下

```math
\hat\beta_j^{OLS}
\sim
N(W_j\gamma,\Lambda_j),
\qquad
\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1}.
```

给定 $`D_k, \sigma_k^2`$，对 $`\gamma`$ 作 GLS 得

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS},
```

其中 $`\Lambda_{jk}=D_k+\sigma_k^2(X_j^TX_j)^{-1}`$。该公式是给定 $`D, \sigma^2`$ 时 $`\gamma`$ 的边际极大似然估计；它与严格 EM 的 $`\gamma`$ 更新在收敛点一致，并显著加快两层模型的计算。

**为何组级 GLS 等于全样本边际 MLE。** 在边际模型 $`y_j \sim N(X_jW_j\gamma, V_j)`$ 且 $`V_j=X_jDX_j^T+\sigma^2 I_{n_j}`$ 下，$`\gamma`$ 的全样本 GLS（即边际 MLE）为

```math
\gamma
=
(\sum_{j}W_j^TX_j^TV_j^{-1}X_jW_j)^{-1}
\sum_{j}W_j^TX_j^TV_j^{-1}y_j.
```

记

```math
\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1},
\qquad
H_j=X_j(X_j^TX_j)^{-1}X_j^T
```

由 Woodbury 展开 $`V_j^{-1}`$ 可证两条恒等式

```math
X_j^TV_j^{-1}X_j=\Lambda_j^{-1},
\qquad
X_j^TV_j^{-1}y_j=\Lambda_j^{-1}\hat\beta_j^{OLS},
```

后者用到投影性质 $`X_j^TV_j^{-1}(I-H_j)=0`$（因 $`X_j^T(I-H_j)=0`$）。代入即得上面对 $`\hat\beta_j^{OLS}`$ 的 GLS 形式。故"对组级 OLS 估计做 GLS"与"对个体数据做 GLS"完全等价，二者都等于给定 $`(D,\sigma^2)`$ 时 $`\gamma`$ 的边际 MLE；而 EM 整体收敛到边际似然的稳定点时，其 $`\gamma`$ 分量恰为该条件 MLE，故两种更新在收敛点一致。

#### 4.5.2 随机效应协方差 $`D`$

记 $`S_{jk}=m_{jk}m_{jk}^T+V_{jk}`$。保留与 $`D`$ 有关的项：

```math
-\frac{J}{2}\log|D|
-\frac{1}{2}\sum_{j=1}^{J}\mathrm{tr}(D^{-1}S_{jk}).
```

使用矩阵求导：

```math
\frac{\partial}{\partial D}\log|D|=D^{-1},
```

```math
\frac{\partial}{\partial D}\mathrm{tr}(D^{-1}S)
=-D^{-1}SD^{-1}.
```

令偏导为零：

```math
-\frac{J}{2}D^{-1}
+\frac{1}{2}\sum_{j=1}^{J}D^{-1}S_{jk}D^{-1}
=0.
```

左右同乘 $`D`$，得 $`JD=\sum_{j=1}^{J}S_{jk}`$。因此（ML 更新）

```math
D_{k+1}^{\mathrm{ML}}
=
\frac{1}{J}
\sum_{j=1}^{J}
(m_{jk}m_{jk}^T+V_{jk}).
```

**ML 的小样本向下偏与 REML 校正。** 上式把 $`\gamma`$ 当作已知（用其估计 $`\hat\gamma`$ 代入）。但 $`\hat\gamma`$ 由数据估计，会"吸收"一部分组间变异，使 $`m_{jk}`$ 围绕 $`W_j\hat\gamma`$ 的离散度偏小，从而 $`D_{k+1}^{\mathrm{ML}}`$ 系统性低估 $`D`$。其量级约为 $`(J-(q+1))/J`$（每个随机效应分量由 $`q+1`$ 个组级系数拟合均值），组数 $`J`$ 较小时尤为明显（$`J=20`$、$`q=3`$ 时因子 $`\approx0.8`$，即低估约 20%）。

REML 通过把 $`\hat\gamma`$ 的不确定性传播回随机效应来消除该偏。记边际 GLS 下 $`\hat\gamma`$ 的协方差

```math
C=\Big(\sum_{j=1}^{J}W_j^T\Lambda_j^{-1}W_j\Big)^{-1},
\qquad
\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1},
```

由 $`\hat\mu_j=DX_j^TV_j^{-1}(y_j-X_jW_j\hat\gamma)`$ 得 $`\partial\hat\mu_j/\partial\hat\gamma=-DX_j^TV_j^{-1}X_jW_j=-D\Lambda_j^{-1}W_j=:-B_j`$（用到恒等式 $`X_j^TV_j^{-1}X_j=\Lambda_j^{-1}`$，见 §4.5.1），故 $`E[\mu_j\mu_j^T]`$ 额外含 $`B_jCB_j^T`$。REML 更新为

```math
D_{k+1}^{\mathrm{REML}}
=
\frac{1}{J}
\sum_{j=1}^{J}
\big(m_{jk}m_{jk}^T+V_{jk}+B_{jk}\,C_k\,B_{jk}^T\big),
\qquad
B_{jk}=D_k\Lambda_{jk}^{-1}W_j.
```

该更新在任意组数 $`J`$ 下都使 $`\hat D`$ 基本无偏；其单调不减的目标量相应地由观测数据边际对数似然改为**限制对数似然** $`\ell_R=\ell_{\mathrm{ML}}(\hat\gamma)+\tfrac12\log|C|+\tfrac{p_\gamma}{2}\log(2\pi)`$、$`p_\gamma=(p+1)(q+1)`$（`get_loglik` 在 REML 下返回该量；附加的 $`\tfrac{p_\gamma}{2}\log(2\pi)`$ 即积掉固定效应得到的标准 Harville 常数）。等价地，可用 REML 投影矩阵写为 $`D_{k+1}=\frac1J\sum_j\big(m_{jk}m_{jk}^T+D_k-D_kX_j^TP_{jj}X_jD_k\big)`$，其中 $`P=V^{-1}-V^{-1}(XW)C(XW)^TV^{-1}`$，$`P_{jj}`$ 为 $`P`$ 对应第 $`j`$ 组的 $`n_j\times n_j`$ 对角块。

> **代码开关。** 实现用全局选项 `options(hlm_reml=TRUE)` 切换 REML（仅本章模拟驱动开启）；默认 `FALSE` 即上面的 ML 更新。第 5 章实证（§7.2）保持 ML，以与 `lme4` 的 ML 结果逐位对照。MCEM 下则改为从膨胀协方差 $`N(m_{jk},\,V_{jk}+B_{jk}C_kB_{jk}^T)`$ 抽样，使样本二阶矩匹配 $`D_{k+1}^{\mathrm{REML}}`$。

#### 4.5.3 一级误差方差 $`\sigma^2`$

保留与 $`\sigma^2`$ 有关的项：

```math
-\frac{N}{2}\log\sigma^2
-\frac{1}{2\sigma^2}
\sum_{j=1}^{J}
[
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T)
].
```

记

```math
R_k(\gamma)
=
\sum_{j=1}^{J}
[
\| y_j-X_jW_j\gamma-X_jm_{jk}\|^2
+\mathrm{tr}(X_jV_{jk}X_j^T)
].
```

则相关项为 $`-\frac{N}{2}\log\sigma^2-\frac{R_k(\gamma)}{2\sigma^2}`$。对 $`\sigma^2`$ 求偏导：

```math
\frac{\partial Q}{\partial\sigma^2}
=
-\frac{N}{2\sigma^2}
+\frac{R_k(\gamma)}{2(\sigma^2)^2}.
```

令其为零，得 $`N\sigma^2=R_k(\gamma)`$。代入更新后的 $`\gamma_{k+1}`$：

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

> **$`\sigma^2`$ 的 REML 校正。** 使 $`D`$ 更新膨胀的 $`\hat\gamma`$ 不确定性（§4.5.2）同样进入残差平方和。后验均值处的组残差 $`r_j=(I-X_jDX_j^TV_j^{-1})(y_j-X_jW_j\hat\gamma)`$ 满足 $`\partial r_j/\partial\hat\gamma=-A_j`$，其中 $`A_j=X_jW_j-X_jD\Lambda_j^{-1}W_j`$（同一个 $`D`$ 同时出现在独立因子与 $`\Lambda_j=D+\sigma^2(X_j^TX_j)^{-1}`$ 中，因为二者都来自同一 $`V_j=X_jDX_j^T+\sigma^2 I`$），故 REML 在 $`R_k`$ 上加 $`\sum_j\mathrm{tr}(A_jCA_j^T)`$。**与 $`D`$ 更新（§4.5.2）在更新前的 $`D_k`$ 处评价 REML 量不同，本 $`\sigma^2`$ 块遵循 §4.6（步骤 4）的 multicycle-ECM 刷新，在刚更新的 $`D_{k+1}`$ 处评价 $`A_{jk}`$ 与 GLS 协方差**——令 $`\Lambda_{j,k+1}=D_{k+1}+\sigma_k^2(X_j^TX_j)^{-1}`$、$`C_{k+1}=(\sum_{j=1}^{J}W_j^T\Lambda_{j,k+1}^{-1}W_j)^{-1}`$，于是
> ```math
> \sigma_{k+1}^{2,\mathrm{REML}}
> =\frac{1}{N}\Big(R_k(\gamma_{k+1})+\sum_{j=1}^{J}\mathrm{tr}(A_{jk}\,C_{k+1}\,A_{jk}^T)\Big),
> \qquad A_{jk}=X_jW_j-X_jD_{k+1}\Lambda_{j,k+1}^{-1}W_j.
> ```
> （这与代码 `em_update_sigma2` 中的 `get_reml_pieces(data, D_{k+1}, \sigma_k^2)` 一致：独立的 $`D_{k+1}`$ 因子与 $`\Lambda^{-1}`$/$`C`$ 量都统一用更新后的 $`D_{k+1}`$；两个下标仅在不动点 $`D_k=D_{k+1}`$ 处重合。）这是一般 LMM（§5）$`\sigma^2`$ REML 校正在"固定设计 $`X_jW_j`$、随机设计 $`X_j`$"下的特例，使手写 REML 的 $`\hat\sigma^2`$ 与 `lme4(REML=TRUE)` 吻合到机器精度；ML 下该项消失、除数仍为 $`N`$。MCEM 下 $`\sigma^2`$ 样本仍用 ML 协方差、该项显式加入，避免重复计数。

### 4.6 两层模型 EM 算法流程

> **§4.5 推导与本节实现的关系（multicycle-ECM）。** §4.5 是教科书式 EM 的推导：在 $`\theta_k`$ 处一次性构造 $`Q`$，再对 $`(\gamma, D, \sigma^2)`$ 同时极大化。本节（以及代码 `two-level-model/utils.R`）实际采用 **multicycle-ECM**（Meng & Rubin, 1993）次序——把 $`\gamma`$ 用边际 GLS 一步求解（ECME 步），并在每个参数块更新后**刷新 E 步**：用 $`\gamma_{k+1}`$ 重算条件期望再更新 $`D`$，再用 $`(\gamma_{k+1}, D_{k+1})`$ 重算条件期望再更新 $`\sigma^2`$。每个条件极大化步都不减观测数据边际对数似然，故与标准 EM 共享同一组不动点。下面的流程严格对应代码实现。

给定初始值 $`\gamma_0, D_0, \sigma_0^2`$、最大迭代次数 $`K`$ 和收敛阈值 $`\varepsilon`$，对 $`k=0,1,2,\dots`$ 重复：

1. E 步：对每个组 $`j=1,\dots,J`$，按当前 $`(\gamma_k, D_k, \sigma_k^2)`$ 计算随机效应条件均值与条件协方差：

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

2. M 步 — 固定效应（边际 GLS / ECME 步，不依赖后验）。令

```math
\hat\beta_j^{OLS}=(X_j^TX_j)^{-1}X_j^Ty_j,
\qquad
\Lambda_{jk}=D_k+\sigma_k^2(X_j^TX_j)^{-1},
```

并更新

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS}.
```

3. 刷新 E 步并更新 $`D`$：用新的 $`\gamma_{k+1}`$（仍用 $`D_k, \sigma_k^2`$）重算条件均值

```math
m'_{jk}
=
(X_j^TX_j+\sigma_k^2D_k^{-1})^{-1}
X_j^T(y_j-X_jW_j\gamma_{k+1}),
```

$`V_{jk}`$ 不依赖 $`\gamma`$，故保持不变，于是

```math
D_{k+1}
=
\frac{1}{J}\sum_{j=1}^{J}(m'_{jk}m_{jk}^{\prime T}+V_{jk}).
```

4. 再次刷新 E 步并更新 $`\sigma^2`$：用 $`\gamma_{k+1}`$ 与**已更新的 $`D_{k+1}`$**（$`\sigma_k^2`$ 不变）重算

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

于是

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

5. 若 $`\gamma, D, \sigma^2`$ 的变化量均低于阈值 $`\varepsilon`$，则停止。

> 说明：第 3、4 步对条件矩的"刷新"即 multicycle-ECM 的体现。若改为始终复用第 1 步基于 $`(\gamma_k, D_k)`$ 的 $`m_{jk}, V_{jk}`$（即 §4.5 的一次性 E 步），则退化为标准 EM；两者不动点相同。

### 4.7 两层模型 MCEM

在 MCEM 中，从第 $`k`$ 次迭代给出的条件分布抽样：

```math
\mu_j^{(k+1,m)}
\sim
N(m_{jk},V_{jk}),
\qquad
m=1,\dots,M.
```

> 与 §4.6 一致，实际实现采用 **multicycle-ECM** 次序，而非抽一组样本反复使用：先用闭式 GLS 更新 $`\gamma`$，再在刷新均值 $`m'_{jk}`$（用 $`\gamma_{k+1}`$）处抽样以更新 $`D`$，然后在 $`m''_{jk}`$（用 $`\gamma_{k+1}, D_{k+1}`$）处**重抽样**以更新 $`\sigma^2`$；确切次序见 §4.8。此处在 $`m_{jk}`$ 处一次抽样的写法与之共享同一组不动点，仅为记号简洁。

Monte Carlo 近似的 Q 函数为

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

#### 4.7.1 $`\gamma`$ 更新

按修订稿，因 $`\gamma`$ 的边际 M 步可解析求解，MCEM 中仍使用 GLS 闭式更新：

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS},
```

```math
\Lambda_{jk}=D_k+\sigma_k^2(X_j^TX_j)^{-1}.
```

因此 Monte Carlo 误差不进入 $`\gamma`$ 的解析更新，主要影响 $`D`$ 与 $`\sigma^2`$。

#### 4.7.2 $`D`$ 更新

保留与 $`D`$ 有关的部分：

```math
-\frac{J}{2}\log|D|
-\frac{1}{2M}
\sum_{m=1}^{M}\sum_{j=1}^{J}
(\mu_j^{(k+1,m)})^TD^{-1}\mu_j^{(k+1,m)}.
```

求导并令零：

```math
-\frac{J}{2}D^{-1}
+\frac{1}{2M}
\sum_{m=1}^{M}\sum_{j=1}^{J}
D^{-1}\mu_j^{(k+1,m)}(\mu_j^{(k+1,m)})^TD^{-1}
=0.
```

得到

```math
D_{k+1}
=
\frac{1}{MJ}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\mu_j^{(k+1,m)}(\mu_j^{(k+1,m)})^T.
```

#### 4.7.3 $`\sigma^2`$ 更新

保留与 $`\sigma^2`$ 有关的部分。记

```math
R_M(\gamma)
=
\sum_{m=1}^{M}\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma-X_j\mu_j^{(k+1,m)}\|^2.
```

则相关项为 $`-\frac{N}{2}\log\sigma^2-\frac{R_M(\gamma)}{2M\sigma^2}`$。对 $`\sigma^2`$ 求偏导并令零：

```math
-\frac{N}{2\sigma^2}
+\frac{R_M(\gamma)}{2M(\sigma^2)^2}
=0,
```

即 $`MN\sigma^2=R_M(\gamma)`$。代入 $`\gamma_{k+1}`$：

```math
\sigma_{k+1}^2
=
\frac{1}{MN}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\| y_j-X_jW_j\gamma_{k+1}-X_j\mu_j^{(k+1,m)}\|^2.
```

### 4.8 两层模型 MCEM 算法流程

与 §4.6 的 EM 流程同构，实现同样按 multicycle-ECM 次序——$`\gamma`$ 先解析更新，其后每更新一个方差分量都用最新参数**重新抽样**。给定初始值 $`\gamma_0, D_0, \sigma_0^2`$、采样量 $`M`$、最大迭代次数 $`K`$ 和阈值 $`\varepsilon`$，对 $`k=0,1,2,\dots`$ 重复：

1. M 步 — 固定效应（边际 GLS 闭式，无需采样）：

```math
\gamma_{k+1}
=
(\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}W_j)^{-1}
\sum_{j=1}^{J}W_j^T\Lambda_{jk}^{-1}\hat\beta_j^{OLS}.
```

2. 在 $`(\gamma_{k+1}, D_k, \sigma_k^2)`$ 下对每组抽 $`M`$ 个随机效应样本 $`\mu_j^{(m)}\sim N(m'_{jk}, V_{jk})`$（$`m'_{jk}`$ 即 §4.6 第 3 步的条件均值），更新 $`D`$：

```math
D_{k+1}
=
\frac{1}{MJ}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\mu_j^{(m)}(\mu_j^{(m)})^T.
```

3. 在 $`(\gamma_{k+1}, D_{k+1}, \sigma_k^2)`$ 下**重新抽样** $`\tilde\mu_j^{(m)}\sim N(m''_{jk}, V'_{jk})`$（$`m''_{jk}, V'_{jk}`$ 即 §4.6 第 4 步基于 $`D_{k+1}`$ 的条件矩），更新 $`\sigma^2`$：

```math
\sigma_{k+1}^2
=
\frac{1}{MN}
\sum_{m=1}^{M}\sum_{j=1}^{J}
\|y_j-X_jW_j\gamma_{k+1}-X_j\tilde\mu_j^{(m)}\|^2.
```

4. 因 E 步为随机近似，实际停止准则通常结合最大迭代次数、参数变化量和多次运行稳定性判断；采样误差使路径可能不严格单调（§2）。

## 5. 一般混合线性模型

一般混合线性模型（Linear Mixed Model, LMM）是两层模型的推广：固定效应设计 $`X_i`$ 与随机效应设计 $`Z_i`$ 可彼此不同、任意指定。两层模型是其特例（取固定设计 $`X_jW_j`$、随机设计 $`X_j`$、$`u_i=\mu_j`$、$`G_0=D`$）。完整推导亦见 `code/linear-mixed-model/lmm_derivation.md`。

### 5.1 模型设定

对 $`i=1,\dots,m`$ 个二级单元（组），

```math
y_i = X_i\beta + Z_i u_i + \varepsilon_i,
\qquad
u_i\sim N(0,G_0),
\quad
\varepsilon_i\sim N(0,\sigma^2 I_{n_i}).
```

其中 $`X_i\in\mathbb{R}^{n_i\times p}`$ 为固定效应设计（含截距列），$`Z_i\in\mathbb{R}^{n_i\times k}`$ 为随机效应设计，$`u_i`$（$`k`$ 维随机效应）与 $`\varepsilon_i`$ 相互独立、各组独立。本实现取 $`Z_i=X_i[,1:k]`$（随机截距 + 前 $`k-1`$ 个协变量的随机斜率，即随机系数模型）。记 $`n=\sum_i n_i`$。待估参数为 $`\beta`$（$`p`$ 维）、$`G_0`$（$`k\times k`$）、$`\sigma^2`$，隐变量为 $`u`$。

写成堆叠矩阵形式，

```math
y=X\beta+Zu+\varepsilon,\qquad
X=\begin{pmatrix}X_1\\\vdots\\X_m\end{pmatrix},\quad
Z=\mathrm{diag}(Z_1,\dots,Z_m),\quad
u=\begin{pmatrix}u_1\\\vdots\\u_m\end{pmatrix}\sim N(0,G),\ \ G=I_m\otimes G_0,
```

其中 $`\varepsilon\sim N(0,\sigma^2 I_n)`$。固定效应设计 $`X`$ 按行堆叠，而随机效应设计 $`Z`$ 为**块对角**——每组的 $`u_i`$ 只作用在自己的块内——故边际协方差 $`V=ZGZ^T+\sigma^2 I_n=\mathrm{diag}(V_1,\dots,V_m)`$ 亦为块对角、各组独立。这正是第 4 章两层模型在对应关系 $`X_i\leftrightarrow X_jW_j`$、$`Z_i\leftrightarrow X_j`$、$`u_i\leftrightarrow\mu_j`$、$`G_0\leftrightarrow D`$、$`\beta\leftrightarrow\gamma`$ 下的特例。

### 5.2 边际分布与随机效应后验

观测数据边际分布：

```math
y_i\sim N(X_i\beta, V_i),
\qquad
V_i=Z_iG_0Z_i^T+\sigma^2 I_{n_i}.
```

将 $`u`$ 视为隐变量，各组独立，随机效应后验为 $`u_i\mid y_i\sim N(u_i^*, V_i^*)`$，由 Woodbury 恒等式化为便于计算的形式：

```math
u_i^*=(Z_i^TZ_i+\sigma^2 G_0^{-1})^{-1}Z_i^T(y_i-X_i\beta),
\qquad
V_i^*=\sigma^2(Z_i^TZ_i+\sigma^2 G_0^{-1})^{-1}.
```

### 5.3 完全数据对数似然与先验

取平坦无信息先验 $`\pi(\beta, G_0, \sigma^2)\propto 1`$，故 MAP 与 MLE 一致，EM 极大化观测数据边际对数似然 $`\log p(y\mid\beta, G_0, \sigma^2)`$。完全数据对数似然中与参数有关的部分为

```math
\ell_c(\beta,G_0,\sigma^2\mid y,u)
\propto
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\|y_i-X_i\beta-Z_iu_i\|^2
-\frac{m}{2}\log|G_0|
-\frac{1}{2}\sum_{i=1}^{m}u_i^TG_0^{-1}u_i.
```

$`\log|G_0|`$ 的系数为 $`-m/2`$，对应 $`G_0`$ 的 M 步除数 $`m`$。

### 5.4 EM 的 E 步与 M 步

E 步用 $`E[u_i\mid y]=u_i^*`$、$`E[u_iu_i^T\mid y]=u_i^*u_i^{*T}+V_i^*`$，得

```math
Q\propto
-\frac{n}{2}\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i}[\|y_i-X_i\beta-Z_iu_i^*\|^2+\mathrm{tr}(Z_iV_i^*Z_i^T)]
-\frac{m}{2}\log|G_0|
-\frac{1}{2}\sum_{i}[u_i^{*T}G_0^{-1}u_i^*+\mathrm{tr}(G_0^{-1}V_i^*)].
```

M 步：

- 固定效应 $`\beta`$（边际 GLS / ECME；严格 EM 解 $`(\sum_i X_i^TX_i)^{-1}\sum_i X_i^T(y_i-Z_iu_i^*)`$ 与之同收敛点）：

```math
\beta_{t+1}=(\sum_{i=1}^{m}X_i^TV_i^{-1}X_i)^{-1}\sum_{i=1}^{m}X_i^TV_i^{-1}y_i,
\qquad
V_i=Z_iG_0Z_i^T+\sigma^2 I_{n_i}.
```

- 随机效应协方差 $`G_0`$（ML ÷$`m`$）：

```math
G_{0,t+1}=\frac{1}{m}\sum_{i=1}^{m}(u_i^*u_i^{*T}+V_i^*).
```

- 误差方差 $`\sigma^2`$（ML ÷$`n`$）：

```math
\sigma^2_{t+1}=\frac{1}{n}\sum_{i=1}^{m}[\|y_i-X_i\beta_{t+1}-Z_iu_i^*\|^2+\mathrm{tr}(Z_iV_i^*Z_i^T)].
```

实现采用与 §4.6 相同的 multicycle-ECM 次序：先更新 $`\beta`$（GLS，不依赖后验），再用 $`(\beta_{t+1}, G_{0,t}, \sigma_t^2)`$ 的后验更新 $`G_0`$，再用 $`(\beta_{t+1}, G_{0,t+1}, \sigma_t^2)`$ 的后验更新 $`\sigma^2`$。

REML 口径下，实现会在 $`G_0`$ 更新中加入固定效应不确定性项 $`B_iCB_i^T`$，在 $`\sigma^2`$ 更新中加入 $`\mathrm{tr}(A_iCA_i^T)`$，其中 $`B_i=G_0Z_i^TV_i^{-1}X_i`$、$`A_i=X_i-Z_iG_0Z_i^TV_i^{-1}X_i`$、$`C=\big(\sum_{i=1}^{m}X_i^TV_i^{-1}X_i\big)^{-1}`$ 为 $`\hat\beta`$ 的边际 GLS 协方差（即 §4.5.2/§4.5.3 在固定设计 $`X_i`$、随机设计 $`Z_i`$、协方差 $`G_0`$ 下的特例）；`get_loglik` 返回限制对数似然 $`\ell_R=\ell_{\mathrm{ML}}(\hat\beta)+\tfrac12\log|C|+\tfrac{p}{2}\log(2\pi)`$、$`p=\dim\beta`$。这样既保留 ML 自由度口径，也提供方差分量的无偏 REML 替代。

### 5.5 MCEM

从 $`u_i\mid y_i\sim N(u_i^*,V_i^*)`$ 抽取 $`M`$ 个样本 $`u_i^{(1)},\dots,u_i^{(M)}`$（同样在每个方差分量更新前用最新参数重新抽样），$`\beta`$ 仍用闭式 GLS，方差分量改用样本近似：

```math
G_{0,t+1}=\frac{1}{mM}\sum_{l=1}^{M}\sum_{i=1}^{m}u_i^{(l)}u_i^{(l)T},
\qquad
\sigma^2_{t+1}=\frac{1}{nM}\sum_{l=1}^{M}\sum_{i=1}^{m}\|y_i-X_i\beta_{t+1}-Z_iu_i^{(l)}\|^2.
```

$`M\to\infty`$ 时各更新依大数定律收敛到对应 EM 更新。

## 6. 数值模拟结论

### 6.1 t 回归模拟结论

**单次 EM：迭代过程与 MSE**（图 3.1 / 3.2 左）

<p align="center">
 <img src="code/t-regression/figures/25_t_em_single_iter.png" width="45%" alt="t 回归 EM 单次执行迭代过程">
 <img src="code/t-regression/figures/26_t_em_single_mse.png" width="45%" alt="t 回归 EM 单次执行 MSE 变化">
</p>

**EM 500 次模拟取平均：迭代过程与 MSE**（图 3.1 / 3.2 右；图 3.6）

<p align="center">
 <img src="code/t-regression/figures/21_t_em_500sim_iter.png" width="45%" alt="t 回归 EM 500 次模拟取平均的迭代过程">
 <img src="code/t-regression/figures/22_t_em_500sim_mse.png" width="45%" alt="t 回归 EM 500 次模拟的 MSE 变化">
</p>

**初始化策略对比（500 次模拟）：保守型 vs 扩展型**（图 3.3）

<p align="center">
 <img src="code/t-regression/figures/27_t_em_normal_init_iter.png" width="45%" alt="t 回归 EM 保守型初始化迭代过程（500 次模拟）">
 <img src="code/t-regression/figures/28_t_em_bold_init_iter.png" width="45%" alt="t 回归 EM 扩展型初始化迭代过程（500 次模拟）">
</p>

**自由度扫描：估计误差与迭代次数**（图 3.4）

<p align="center">
 <img src="code/t-regression/figures/29_t_df_error.png" width="45%" alt="t 回归参数估计误差随自由度变化">
 <img src="code/t-regression/figures/30_t_df_iter.png" width="45%" alt="t 回归 EM 迭代次数随自由度变化">
</p>

**单次 MCEM：M=1 与 M=100 采样对比**（图 3.5）

<p align="center">
 <img src="code/t-regression/figures/31_t_mcem_M1_single_iter.png" width="45%" alt="t 回归 MCEM(M=1) 单次执行迭代过程">
 <img src="code/t-regression/figures/32_t_mcem_M100_single_iter.png" width="45%" alt="t 回归 MCEM(M=100) 单次执行迭代过程">
</p>

**MCEM(M=100) 500 次模拟取平均：迭代过程与 MSE**（图 3.6）

<p align="center">
 <img src="code/t-regression/figures/23_t_mcem_500sim_iter.png" width="45%" alt="t 回归 MCEM(M=100) 500 次模拟取平均的迭代过程">
 <img src="code/t-regression/figures/24_t_mcem_500sim_mse.png" width="45%" alt="t 回归 MCEM(M=100) 500 次模拟的 MSE 变化">
</p>

> 图由 `code/t-regression/driver_compare500.R`（21–24）、`driver_em_plots.R`（25–30）、`driver_mcem_plots.R`（31–32）生成（均 `set.seed(2025)`），保存于 `code/t-regression/figures/`。自由度扫描（29/30）为控制耗时采用粗网格，趋势与论文一致。

500 次模拟均值对比（$`n=500`$，$`p=2`$，$`\nu=10`$，真值 $`\beta=(2,3,5)`$、$`\sigma^2=0.4`$；固定初值 $`\beta_0=0`$、$`\sigma_0^2=0.1`$）：

| 指标 | EM | MCEM $`M=100`$ |
|---|---:|---:|
| $`\hat\beta`$ | (2.0005, 3.0008, 5.0000) | (1.9975, 3.0000, 5.0014) |
| $`\hat\sigma^2`$ | 0.3977 | 0.3957 |
| $`\beta`$ 的 MSE | 8.62e−07 | 8.39e−06 |
| $`\sigma^2`$ 的 MSE | 5.50e−06 | 1.89e−05 |
| 总 MSE | 6.36e−06 | 2.73e−05 |
| 平均迭代次数 | 13.49 | 100.00（达上限）|
| 总耗时（秒） | 11.0 | 336.7 |

主要结论：

- t 回归 EM 本质是迭代加权最小二乘，大残差观测会被自动降权。
- 当算法设定自由度与数据真实自由度匹配时，$`\sigma^2`$ 估计更准确。
- $`\beta`$ 对自由度设定相对不敏感，整体估计较稳定。
- $`M`$ 足够大时，MCEM 的估计结果接近解析 EM；$`M`$ 较小时路径会受采样误差影响。

#### 6.1.1 与调库结果的对比与评估

为独立验证手写实现，`code/t-regression/driver_library_compare.R`（`set.seed(2025)`，$`n=500`$、$`\nu=10`$、500 次模拟）把手写 EM/MCEM 与三种现成实现逐一对照：`optim` 直接极大化边际 t 似然（纯 MLE）、第三方 t 回归库 `hett::tlm`、稳健回归 `MASS::rlm`（Huber M-估计，不同损失函数）。

**单数据集逐位对照**：手写 EM 与"`optim` 极大化同一对数后验（含 $`\pi\propto1/\sigma^2`$ 先验）"逐位一致（$`\max|\Delta\beta|=2\times10^{-8}`$，$`|\Delta\sigma^2|=4\times10^{-8}`$）；$`\beta`$ 与 `optim`（纯 MLE）、`hett::tlm` 一致到 $`\approx10^{-5}`$；$`\mathrm{cor}(\beta_{\mathrm{EM}},\beta_{\mathrm{rlm}})=\mathrm{cor}(\beta_{\mathrm{EM}},\beta_{\mathrm{hett}})=1.0000`$。

**500 次模拟均值对比**（真值 $`\beta=(2,3,5)`$、$`\sigma^2=0.4`$）：

| 方法 | $`\hat\beta`$ 均值 | $`\hat\sigma^2`$ | 平均单次 MSE † | 口径说明 |
|---|---|---:|---:|---|
| 手写 EM（MAP）| (2.0005, 3.0008, 5.0000) | 0.3977 | 3.69e−03 | 先验 $`\pi\propto1/\sigma^2`$，$`\sigma^2`$ 除 $`n+2`$ |
| `optim`（边际 MLE）| (2.0005, 3.0008, 5.0000) | 0.3997 | 3.69e−03 | 纯似然，$`\sigma^2`$ 除 $`n`$ |
| `hett::tlm` | (2.0005, 3.0008, 5.0000) | 0.3997 | 3.69e−03 | 第三方 t 回归 MLE 库 |
| `MASS::rlm` | (2.0006, 3.0008, 5.0001) | — | 2.82e−03 | Huber M-估计（仅 $`\beta`$）|

> † **MSE 口径说明。** 本表"平均单次 MSE"是 500 次模拟中**每次**估计误差平方和的均值（$`\frac{1}{S}\sum_s[\,\lVert\hat\beta_s-\beta\rVert^2+(\hat\sigma^2_s-\sigma^2)^2\,]`$），衡量单次估计精度；而 §6.1 主表的"总 MSE"（6.36e−06）是**先对 500 次估计取平均、再算偏差**（$`\lVert\bar{\hat\beta}-\beta\rVert^2+\dots`$），衡量估计量的偏倚。二者度量不同、数值不可直接比较；两口径下手写与各库均一致。

> **关于 $`\sigma^2`$ 的口径差异。** 手写 EM 的 $`\hat\sigma^2=0.3977`$ 略低于 `optim`/`hett` 的 $`0.3997`$，差异恰为 MAP（除 $`n+2`$）与纯 MLE（除 $`n`$）之比 $`\approx1-2/n`$，**并非实现错误**；$`\beta`$ 不受先验影响，三者逐位一致。这印证了 §3.2、§3.5 中"先验 $`\pi\propto1/\sigma^2`$ 把 ML 改为 MAP、除数由 $`n`$ 变为 $`n+2`$"的推导是自洽且正确的。

### 6.2 两层线性模型模拟结论

设置 $`J=20`$ 组、$`p=2`$、$`q=3`$，组内样本量 $`n_j\in\{60,80,100\}`$。真实参数按论文 §4.1.2 的设定在 `set.seed(2025)` 下随机生成：$`D=Q\Lambda Q^{\top}`$，其中 $`Q`$ 为随机正交因子（高斯矩阵的 QR 正交因子），固定谱 $`\Lambda=\mathrm{diag}(5,6,7)`$——即特征值为 $`(5,6,7)`$ 的随机旋转矩阵；固定效应 $`\gamma`$ 与一级误差方差 $`\sigma^2`$ 取自均匀分布（$`\gamma_i\sim U(0,12)`$、$`\sigma^2\sim U(1,5)`$）。固定初值 $`\gamma_0=0.1`$、$`\sigma_0^2=0.1`$。方差分量 $`D`$ 采用 **REML** 估计（`options(hlm_reml=TRUE)`，见 §4.5.2）。下列图分别由 `driver_em_plots.R`、`driver_mcem_plots.R`、`driver_compare500.R`（均 `set.seed(2025)`）生成，保存于 `code/two-level-model/figures/`。（下方的库对照验证使用对角为 $`(5,6,7)`$ 的固定 $`D`$，以便与 `lme4` 做受控比较。）

> **为何用 REML。** $`D`$ 的极大似然（ML）估计在小组数下有约 $`(J-(q+1))/J`$ 的固有向下偏：$`J=20`$、$`q=3`$ 时因子 $`\approx0.8`$，曾使 $`\hat D`$ 对角系统性低于真值约 20%（$`\hat\sigma^2`$ 不受影响；这并非实现错误，ML 估计量相合且 §7.2 与 `lme4` ML 逐位吻合）。改用 REML（把 $`\hat\gamma`$ 的不确定性传播回随机效应，每组加校正项 $`B_jCB_j^T`$，推导见 §4.5.2）后，$`\hat D`$ 在 $`J=20`$ 的小组数下即基本无偏（对角恢复到真值约 100%）。实证 §7.2 仍用 ML 以对照 `lme4` ML。

**单次 EM 迭代过程与 MSE**

<p align="center">
 <img src="code/two-level-model/figures/01_em_single_iter.png" width="45%" alt="单次 EM 迭代过程">
 <img src="code/two-level-model/figures/02_em_single_mse.png" width="45%" alt="单次 EM 的 MSE 变化">
</p>

**EM 500 次模拟取平均**

<p align="center">
 <img src="code/two-level-model/figures/03_em_500sim_iter.png" width="45%" alt="EM 500 次模拟取平均的迭代过程">
 <img src="code/two-level-model/figures/04_em_500sim_mse.png" width="45%" alt="EM 500 次模拟取平均的 MSE 变化">
</p>

**初始化策略稳健性（上：同分布型；下：扩展型）**

<p align="center">
 <img src="code/two-level-model/figures/05_em_normal_init_iter.png" width="45%" alt="同分布型初始化迭代过程">
 <img src="code/two-level-model/figures/06_em_normal_init_mse.png" width="45%" alt="同分布型初始化 MSE 变化">
</p>
<p align="center">
 <img src="code/two-level-model/figures/07_em_bold_init_iter.png" width="45%" alt="扩展型初始化迭代过程">
 <img src="code/two-level-model/figures/08_em_bold_init_mse.png" width="45%" alt="扩展型初始化 MSE 变化">
</p>

**单次 MCEM 迭代过程与 MSE（上：M=10；下：M=100）**

<p align="center">
 <img src="code/two-level-model/figures/11_mcem_M10_single_iter.png" width="45%" alt="单次 MCEM(M=10) 迭代过程">
 <img src="code/two-level-model/figures/12_mcem_M10_single_mse.png" width="45%" alt="单次 MCEM(M=10) 的 MSE 变化">
</p>
<p align="center">
 <img src="code/two-level-model/figures/13_mcem_M100_single_iter.png" width="45%" alt="单次 MCEM(M=100) 迭代过程">
 <img src="code/two-level-model/figures/14_mcem_M100_single_mse.png" width="45%" alt="单次 MCEM(M=100) 的 MSE 变化">
</p>

**MCEM 500 次模拟取平均（上：M=10；下：M=100）**

<p align="center">
 <img src="code/two-level-model/figures/15_mcem_M10_500sim_iter.png" width="45%" alt="MCEM(M=10) 500 次模拟取平均的迭代过程">
 <img src="code/two-level-model/figures/16_mcem_M10_500sim_mse.png" width="45%" alt="MCEM(M=10) 500 次模拟取平均的 MSE 变化">
</p>
<p align="center">
 <img src="code/two-level-model/figures/17_mcem_M100_500sim_iter.png" width="45%" alt="MCEM(M=100) 500 次模拟取平均的迭代过程">
 <img src="code/two-level-model/figures/18_mcem_M100_500sim_mse.png" width="45%" alt="MCEM(M=100) 500 次模拟取平均的 MSE 变化">
</p>

**EM 与 MCEM(M=50) 500 次模拟对比**

<p align="center">
 <img src="code/two-level-model/figures/51_hlm_em_500sim_iter.png" width="45%" alt="两层模型 EM 500 次模拟取平均的迭代过程">
 <img src="code/two-level-model/figures/52_hlm_em_500sim_mse.png" width="45%" alt="两层模型 EM 500 次模拟的 MSE 变化">
</p>
<p align="center">
 <img src="code/two-level-model/figures/53_hlm_mcem_500sim_iter.png" width="45%" alt="两层模型 MCEM(M=50) 500 次模拟取平均的迭代过程">
 <img src="code/two-level-model/figures/54_hlm_mcem_500sim_mse.png" width="45%" alt="两层模型 MCEM(M=50) 500 次模拟的 MSE 变化">
</p>

500 次模拟均值对比（$`J=20`$、REML，`max_iter=30`，由 `driver_compare500.R` 输出；真实 $`D`$ 特征值为 $`(5,6,7)`$，随机抽得 $`\sigma^2\approx2.693`$）。$`D`$ 的 MSE 为 Frobenius 误差 $`\lVert\hat D-D\rVert_F`$（旋转不变）：

| 指标 | EM | MCEM $`M=50`$ |
|---|---:|---:|
| $`\hat\sigma^2`$（真值 $`\approx2.693`$） | 2.6891 | 2.6900 |
| $`\gamma`$ 的 MSE | 0.16810 | 0.16562 |
| $`D`$ 的 MSE | 0.28075 | 0.19965 |
| $`\sigma^2`$ 的 MSE | 0.00369 | 0.00282 |
| 总 MSE | 0.45254 | 0.36808 |
| 平均迭代次数 | 11.58 | 30.00（达上限）|
| 总耗时（秒） | 61.1 | 320.1 |

> 对照：同设置下若用 ML（`options(hlm_reml=FALSE)`），$`\hat D`$ 特征值约 $`(4.0,4.8,5.7)\approx0.8\times(5,6,7)`$、$`D`$ 的 MSE $`\approx2.2`$——即被低估约 20% 的旧结果；REML 把 $`D`$ 的 MSE 降到 $`\approx0.21`$（约 10 倍）。

主要结论：

- 采用 REML 后，EM 单调增加**限制对数似然**（§4.5.2）；$`\gamma`$、$`\sigma^2`$、$`D`$ 均准确恢复真值。
- 在 $`J=20`$ 的小组数下，$`\hat D`$ 特征谱恢复到真值约 100%（ML 会低估约 20%），$`D`$ 不再主导总误差——REML 把 $`\hat\gamma`$ 的不确定性传播回随机效应，消除了方差分量的小样本向下偏。
- EM 与 MCEM（$`M=50`$）精度相当，方差分量估计互有高低、均接近真值。
- MCEM 因 Monte Carlo 噪声在严格容差下常达到最大迭代次数（本设置 30 步），且每步重抽样使其显著更慢（$`\approx5.2\times`$），但参数估计仍接近 EM。

#### 6.2.1 与调库结果的对比与评估

`code/two-level-model/driver_library_compare.R`（`set.seed(2025)`，$`J=20`$、200 次模拟）把手写 EM/MCEM 与金标准 `lme4::lmer` 在同一批数据上对照，且**两种口径分别对齐**：手写 EM(ML) ↔ `lmer(REML=FALSE)`，手写 EM(REML) ↔ `lmer(REML=TRUE)`。

**单数据集逐位对照**：
- ML：手写 EM 与 `lme4`(ML) $`\max|\Delta\gamma|=10^{-9}`$、$`\max|\Delta D|=2\times10^{-6}`$、对数似然差 $`2\times10^{-12}`$。
- REML：手写 EM 与 `lme4`(REML) $`\max|\Delta\gamma|=2\times10^{-9}`$、$`\max|\Delta D|=4\times10^{-6}`$、$`|\Delta\sigma^2|=3\times10^{-8}`$、对数似然差 $`4\times10^{-12}`$，确证**两项** REML 校正——$`D`$ 的 $`B_jCB_j^T`$（§4.5.2）与 $`\sigma^2`$ 的 $`\mathrm{tr}(A_jCA_j^T)`$（§4.5.3）——均推导正确。

**200 次模拟（偏差/MSE 评估，真值 $`D`$ 对角 $`(5,6,7)`$、$`\sigma^2=3`$）**：

| 方法 | $`\hat D`$ 对角 | $`\hat\sigma^2`$ | 与对应 `lme4` 平均 max-diff |
|---|---|---:|---|
| 手写 EM(ML) | (4.076, 4.950, 5.718) | 2.986 | $`3\times10^{-6}`$（vs `lme4` ML）|
| 手写 EM(REML) | (5.104, 6.197, 7.158) | 2.986 | $`6\times10^{-6}`$（vs `lme4` REML）|

ML 的 $`\hat D`$ 对角约为真值的 $`(J-(q+1))/J=16/20=0.80`$ 倍——观测值 $`(4.08,4.95,5.72)\approx0.80\times(5,6,7)`$，与 §4.5.2 的偏差量级完全吻合；REML 校正后恢复到真值的 $`\approx102\%`$。两种口径下手写实现都与 `lme4` 逐位一致，**证明 ML 向下偏是估计量固有性质而非实现缺陷**。（本 200 次模拟验证使用对角为 $`(5,6,7)`$ 的固定 $`D`$——为与 `lme4` 做受控比较，区别于 §6.2 主模拟所用的 QR-随机真实 $`D`$。）

### 6.3 一般混合线性模型模拟结论

参数设置如下（seed=2025，500 次模拟）：

```math
m=50,\quad n_i\in\{15,20,25\},\quad \beta=(1,2,-1,0.5),\quad
G_0=\begin{pmatrix}4&1\\1&2\end{pmatrix},\quad \sigma^2=1.
```

<p align="center">
 <img src="code/linear-mixed-model/figures/41_lmm_em_500sim_iter.png" width="45%" alt="一般混合线性模型 EM 500 次模拟平均迭代过程">
 <img src="code/linear-mixed-model/figures/42_lmm_mcem_M200_500sim_iter.png" width="45%" alt="一般混合线性模型 MCEM(M=200) 500 次模拟平均迭代过程">
</p>

> 图为 **500 次模拟平均**的迭代轨迹——对独立数据集取平均会抵消单数据集的抽样噪声，故曲线收敛到真值、与下表均值一致（单次轨迹则不会）。由 `code/linear-mixed-model/driver_compare500.R`（`set.seed(2025)`）生成，保存于 `code/linear-mixed-model/figures/`。下表给出 EM 与 MCEM 在 Monte Carlo 采样数 $`M\in\{20,50,100,200\}`$ 下、均按 500 次模拟取平均的结果。

| 指标 | EM | MCEM $`M=20`$ | MCEM $`M=50`$ | MCEM $`M=100`$ | MCEM $`M=200`$ |
|---|---:|---:|---:|---:|---:|
| $`\hat\beta`$ | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) |
| $`\hat G_0`$ 对角 | (3.921, 1.953) | (3.920, 1.953) | (3.921, 1.953) | (3.921, 1.952) | (3.921, 1.952) |
| $`\hat\sigma^2`$ | 0.9950 | 0.9947 | 0.9951 | 0.9950 | 0.9950 |
| 总 MSE | 0.13292 | 0.13329 | 0.13214 | 0.13369 | 0.13305 |
| 平均迭代次数 | 8.34 | 50.00（达上限）| 50.00（达上限）| 50.00（达上限）| 50.00（达上限）|
| 总耗时（秒）† | 30.6 | 441 | 824 | 1525 | 2839 |

> † 五个配置都在**同一批** 500 个数据集上运行（配对对比），故 MCEM 各列之间唯一变化的就是 Monte Carlo 采样数 $`M`$。耗时为顺序执行的单配置墙钟时间（每个配置的计算量与所用数据无关）。

主要结论：

- 在同一批数据上，EM 与各 $`M`$ 的 MCEM 都准确恢复真值，且估计几乎完全一致：$`\hat\beta`$ **完全相同**（它是闭式 GLS 解，不受 Monte Carlo E 步影响），$`\hat G_0`$、$`\hat\sigma^2`$ 仅相差极小的 Monte Carlo 噪声。即便 $`M=20`$，MCEM 也已复现 EM 的估计；更大的 $`M`$ 只是把迭代轨迹进一步平滑（$`M=200`$ 的轨迹最干净）。
- $`\hat G_0`$ 对角略偏小（$`\approx(3.92, 1.95)`$ vs $`(4, 2)`$）是方差分量极大似然的固有向下偏（随组数 $`m`$ 增大缓解；§6.3.1 的 REML 可消除）；非对角项在单个数据集上有噪声，500 次平均后准确恢复 $`\approx1`$。
- MCEM 迭代达 50 次上限，耗时随 $`M`$ 近似线性增长——从 $`\approx14\times`$（$`M=20`$）到 $`\approx93\times`$（$`M=200`$）的 EM 墙钟时间——在此**无精度收益**，因为 E 步本就有闭式解。MCEM 仅在 E 步不可解析时才划算。

#### 6.3.1 与调库结果的对比与评估

`code/linear-mixed-model/driver_library_compare.R`（`set.seed(2025)`，$`m=50`$、200 次模拟）把手写 EM/MCEM 与两大标准库 `lme4::lmer`、`nlme::lme` 在 ML 与 REML 双口径下对照。每行均对齐相同目标：手写 ML ↔ 库 ML，手写 REML ↔ 库 REML。

**单数据集逐位对照**：手写 EM 与 `lme4`、`nlme` 在相同口径下逐位一致——ML 的方差分量差异约 $`10^{-7}`$ 量级；补入固定效应自由度校正后，REML 也达到同一量级。

**200 次模拟均值对比**（真值 $`\beta=(1,2,-1,0.5)`$、$`G_0`$ 对角 $`(4,2)`$、$`\sigma^2=1`$）：

| 方法 | $`\hat\beta`$ 均值 | $`\hat G_0`$ 对角 | $`\hat\sigma^2`$ | 平均单次 MSE † |
|---|---|---|---:|---:|
| 手写 EM | (0.990, 1.989, −0.996, 0.504) | (3.987, 1.952) | 0.9976 | 1.2854 |
| `lme4`(ML) | (0.990, 1.989, −0.996, 0.504) | (3.987, 1.952) | 0.9976 | 1.2854 |

> † 与 §6.1.1 思路类似，但**口径不同**：此处"平均单次 MSE"为各次模拟误差*范数* $`\lVert\hat\beta-\beta\rVert_2+\lVert\hat G_0-G_0\rVert_F+|\hat\sigma^2-\sigma^2|`$ 的均值（方差分量模型的 `get_mse` 口径），而 §6.1.1 累加的是*平方*误差——故两表的单次列数值不可直接比较。它衡量单次估计精度；§6.3 主表的"总 MSE"（0.13292）则是对 500 次估计先平均、再取偏差范数，度量的是估计量偏差——二者不可直接比较。

手写 EM 与 `lme4`(ML) 在 200 次模拟上平均 $`\max`$-diff $`=4.8\times10^{-7}`$，与 `nlme::lme` 同样逐位一致，**确证手写 EM 即标准 LMM 的极大似然估计**。若 ML 方差分量不理想，同一驱动脚本现已并列报告 REML；REML 路径使用 `options(lmm_reml=TRUE)`，并与 `lme4(REML=TRUE)` / `nlme(method="REML")` 对齐。MCEM 的 $`\beta`$ 由闭式 GLS 给出（故各 $`M`$ 下完全相同，见 §6.3 主表）；仅方差分量含蒙特卡洛噪声，且随 $`M`$ 增大而减小，但始终接近真值。

## 7. 实证分析（深度学习真实数据）

对应论文第 5 章。两条流水线把第 3、4 章的 EM/MCEM 实现**直接复用**到真实数据（仅替换数据），完整复现说明见 `code/empirical_README.md`。随机种子统一 `20250529`。

### 7.1 Student's t 回归 — UTKFace 人脸年龄

- **数据与任务**：Hugging Face `py97/UTKFace-Cropped` 取 8000 张人脸，ResNet-50 冻结嵌入（2048 维）→ PCA 降至 30 维，回归年龄。$`n=8000`$，预测变量 31（含截距）。
- **方法**：自由度 $`\nu`$ 未知，在网格 $`\{1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50\}`$ 上用 profile 边际似然选 $`\nu^*`$（在 EM/MAP 拟合处评价，其方差与纯 MLE 仅差 $`n/(n+2)`$ 因子）；$`\nu^*`$ 下跑 EM 与 MCEM（$`M=20/50/200`$）；并以 `optim` 直接极大化边际 t 似然、`MASS::rlm` 双重独立验证。为支持 $`n=8000`$，实现用按行缩放 $`wX`$ 替代 $`\mathrm{diag}(w)`$（与 §3.5 加权最小二乘数学等价）。
- **关键结果**：$`\nu^*=6`$；t 边际对数似然 $`-32029.9`$，较正态基线（OLS，$`-32180.9`$）**提升 151.0**；EM(MAP) 与 `optim` 边际 MLE 固定效应系数最大差 $`0.00019`$，EM 与 `rlm` 系数相关 $`\approx1.0`$；E 步权重 $`\gamma_i`$ 随残差绝对值单调下降，被最强下调的 10% 样本中年龄 $`\ge 60`$ 占 $`59.5\%`$（全样本 $`11.3\%`$），即高龄稀疏样本被自动降权。

| 系数 | OLS | EM(t) | MCEM $`M=20`$ | MCEM $`M=200`$ | optim |
|---|---:|---:|---:|---:|---:|
| 截距 | 33.276 | 32.410 | 32.426 | 32.402 | 32.410 |
| PC1 | 2.383 | 2.444 | 2.468 | 2.445 | 2.444 |
| PC2 | 10.648 | 10.687 | 10.658 | 10.685 | 10.687 |

图由 `code/empirical-t-utkface/code/03_replot.R` 生成，保存于 `code/empirical-t-utkface/output/`：

<p align="center">
 <img src="code/empirical-t-utkface/output/71_qq_normal.png" width="45%" alt="OLS 残差正态 QQ 图（端部偏离示重尾）">
 <img src="code/empirical-t-utkface/output/72_profile_nu.png" width="45%" alt="自由度 ν 的 profile 边际似然">
</p>
<p align="center">
 <img src="code/empirical-t-utkface/output/73_weights.png" width="45%" alt="E 步权重随残差单调下调">
 <img src="code/empirical-t-utkface/output/74_qq_t.png" width="45%" alt="t(ν*) 分位-分位图（重尾被模型吸收）">
</p>

### 7.2 两层线性模型 — CIFAR-10H 人类反应时

- **数据与任务**：CIFAR-10H 标注者对 CIFAR-10 测试图的反应时。一级 $`\log RT_{ij}=b_{0j}+b_{1j}\,\mathrm{diff}_{ij}+b_{2j}\,\mathrm{correct}_{ij}+b_{3j}\,\mathrm{trial}_{ij}+\varepsilon_{ij}`$（协变量分别为难度、正确、试次；其中难度与试次在拟合前已 z 标准化为均值 0、标准差 1，故 $`b_{1j}`$、$`b_{3j}`$ 为每标准差效应，正确为二值），二级 $`\beta_j \sim N(\gamma, D)`$（标注者随机系数，$`q=0`$、$`W_j=I_4`$）。剔除组内设计阵秩亏（正确率 0 或 100%）的标注者后 $`J=299`$，$`N=59699`$。
- **方法**：**直接复用**第 4 章两层模型 EM/MCEM（仅覆盖 `get_X_j/y_j/W_j` 访问器以按组读数，避免巨型块对角阵）；跑 EM 与 MCEM（$`M=20/50/200`$）；以 `lme4`（ML）金标准对照。
- **关键结果**：固定效应 $`\gamma`$ =（截距 7.648、难度 0.130、正确 −0.257、试次 −0.033）；$`\sigma^2=0.1191`$ 与 `lme4` 完全一致，边际对数似然 $`-22423.06`$ 与 `lme4` **逐位吻合**（差 $`\lt 0.1`$）；随机效应 SD 为 $`(0.327, 0.050, 0.220, 0.038)`$；截距 ICC $`=0.473`$（标注者间 baseline 速度差异占比）；MCEM（各 $`M`$）与 EM 吻合到约 4 位小数；EM 边际对数似然在真实数据上单调上升。

| 固定效应 | EM | MCEM $`M=20`$ | MCEM $`M=200`$ | lme4 |
|---|---:|---:|---:|---:|
| 截距 | 7.6477 | 7.6475 | 7.6478 | 7.6477 |
| 难度 | 0.1299 | 0.1299 | 0.1299 | 0.1299 |
| 正确 | −0.2572 | −0.2569 | −0.2572 | −0.2572 |
| 试次 | −0.0334 | −0.0334 | −0.0334 | −0.0334 |

图由 `code/empirical-2level-cifar10h/code/03_replot.R` 生成，保存于 `code/empirical-2level-cifar10h/output/`：

<p align="center">
 <img src="code/empirical-2level-cifar10h/output/61_loglik_monotone.png" width="45%" alt="EM 边际对数似然在真实数据上单调上升">
 <img src="code/empirical-2level-cifar10h/output/62_caterpillar_intercept.png" width="45%" alt="标注者随机截距 caterpillar 图">
</p>
<p align="center">
 <img src="code/empirical-2level-cifar10h/output/63_gamma_convergence.png" width="45%" alt="固定效应 γ 的 EM 收敛轨迹">
</p>

### 7.3 实证总结

已记录的模拟与实证实验支持三条观察：EM 的目标函数上升、MCEM 的采样量与精度/成本存在权衡，以及足够采样时 MCEM 与 EM 的估计接近。t 回归与 `optim`/`rlm`、两层模型与 `lme4` 的数值对照提供实现核验；具体差值与口径见上表，不概括为所有方法逐位一致，也不构成任意数据上的正确性证明。

**模拟与实证的调库对照。** 模拟端见 §6.1.1（`optim`/`hett::tlm`/`MASS::rlm`）、§6.2.1（`lme4` 的 ML 与 REML）、§6.3.1（`lme4`/`nlme` 的 ML 与 REML），可用各模型的 `driver_library_compare.R` 复现；实证端见 §7.1 与 §7.2。比较前必须统一目标函数：t 回归 MAP 的 $`n+2`$ 分母及 ML/REML 方差估计并非同一口径；有限采样的 MCEM 还带有 Monte Carlo 误差，不应统一宣称达到机器精度。

## 8. 仓库结构

```text
.
├── README.md                     # 本文档：EM/MCEM 推导 + 模型应用 + 实证结果
├── Algorithms.tex                  # 论文 LaTeX 源；Algorithms.pdf 为已纳入版本控制的编译产物（由 xelatex 重新生成）
├── LICENSE                         # MIT 许可证（Copyright (c) 2026 Li Xiuyin）
└── code/
    ├── README.md                 # 三模型代码总览与公共约定
    ├── docs/                      # code_review_report.md —— 对抗式代码复核（27 条发现 / 21 条确认）
    ├── empirical_README.md       # 两条实证流水线复现说明
    ├── general/                  # make_em_concept.R —— EM 下界示意图（图 2.1）
    ├── t-regression/             # 第 3 章：utils.R / em.R / mcem.R / driver_*.R / README.md
    ├── two-level-model/          # 第 4 章：同上 + figures/
    ├── linear-mixed-model/       # 第 4 章扩展：同上 + lmm_derivation.md
    ├── empirical-t-utkface/      # code/ data/ output/（UTKFace t 回归）
    └── empirical-2level-cifar10h/ # code/ data/ output/（CIFAR-10H 两层模型）
```

代码与文档入口：

| 路径 | 内容 |
|---|---|
| `code/README.md` | 三模型代码总览、公共约定、运行方式、结论速览 |
| `code/t-regression/` | t 回归 EM/MCEM 模拟（`README.md` 含迭代公式与验证结论）|
| `code/two-level-model/` | 两层线性模型 EM/MCEM 模拟（`README.md`）|
| `code/linear-mixed-model/` | 一般 LMM EM/MCEM 模拟；`lmm_derivation.md` 为完整推导 |
| `code/empirical-t-utkface/` | UTKFace 深度嵌入 t 回归实证 |
| `code/empirical-2level-cifar10h/` | CIFAR-10H 反应时两层模型实证 |
| `code/docs/code_review_report.md` | 三套代码的对抗式复核报告 |

## 参考文献

1. Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Wu, C. F. J. (1983). On the convergence properties of the EM algorithm.
3. Wei, G. C. G., and Tanner, M. A. (1990). A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms.
4. Liu, C., and Rubin, D. B. (1995). ML estimation of the t distribution using EM and its extensions.

## 许可证

基于 MIT 许可证发布 —— 参见 [LICENSE](LICENSE)（Copyright (c) 2026 Li Xiuyin）。
