> [English](README.md) | **中文**

# 一般混合线性模型（LMM）的 EM / MCEM 估计

两层线性模型的推广：固定效应设计 $X$ 与随机效应设计 $Z$ 可彼此不同、任意指定。

## 模型

$$
y_i = X_i\beta + Z_i u_i + \varepsilon_i,\qquad u_i\sim\mathcal N(0,G_0),\quad \varepsilon_i\sim\mathcal N(0,\sigma^2 I_{n_i}),\quad i=1,\dots,m,
$$

$X_i$ 为 $n_i\times p$ 固定设计（含截距），随机设计取 $Z_i=X_i[\,,1{:}k]$（随机截距 + 前 $k-1$ 个协变量的随机斜率）。
待估参数：$\beta$（$p$ 维）、$G_0$（$k\times k$ 随机效应协方差）、$\sigma^2$。隐变量为随机效应 $u$。

**先验**：平坦无信息先验 $\pi(\beta,G_0,\sigma^2)\propto 1$ ⇒ ML 口径下 $G_0$ 的 M 步除数为组数 $m$（MCEM 为 $M\!\cdot\!m$），$\sigma^2$ 除以 $n$。
此约定下 `get_loglik`（边际对数似然）即 EM 单调不减的目标。

当组数较小或中等时，ML 方差分量可能因固定效应消耗自由度而明显偏小。设置 `options(lmm_reml=TRUE)` 可启用 REML 校正：代码会把 GLS 固定效应估计的不确定性传播到 $G_0$ 与 $\sigma^2$ 更新中，并将 `get_loglik` 切换为限制对数似然。

完整推导见同目录 **`lmm_derivation.md`**。

## 迭代公式

- **E 步**：$u_i^{*}=(Z_i^{T}Z_i+\sigma^2 G_0^{-1})^{-1}Z_i^{T}(y_i-X_i\beta)$，$V_i^{*}=\sigma^2(Z_i^{T}Z_i+\sigma^2 G_0^{-1})^{-1}$。
- **$\beta$（闭式 GLS）**：$\hat\beta=(\sum_i X_i^{T}V_i^{-1}X_i)^{-1}\sum_i X_i^{T}V_i^{-1}y_i$，$V_i=Z_iG_0Z_i^{T}+\sigma^2 I$。
- **$G_0$（ML ÷$m$）**：$\hat G_0=\frac1m\sum_i(u_i^{*}u_i^{*T}+V_i^{*})$。
- **$\sigma^2$（ML ÷$n$）**：$\hat\sigma^2=\frac1n\sum_i[\lVert y_i-X_i\beta-Z_i u_i^{*}\rVert^2+\mathrm{tr}(Z_iV_i^{*}Z_i^{T})]$。
- **REML 选项**：在方差更新中加入固定效应不确定性校正项，并使用限制对数似然；通过 `options(lmm_reml=TRUE)` 启用。
- **MCEM**：$\beta$ 仍闭式 GLS；$G_0=\frac{1}{mM}\sum_l\sum_i u_i^{(l)}u_i^{(l)T}$，$\sigma^2=\frac{1}{nM}\sum_l\sum_i\lVert\cdot\rVert^2$。

## 文件

| 文件 | 说明 |
|---|---|
| `data_generation.R` | `generate_lmm_data`、`random_init`/`bold_random_init` |
| `utils.R` | `get_post_u`、`em_update_beta/G0/sigma2`、`mcem_update_*`、`get_loglik`、`get_mse`、`run_single_em_or_mcem`、`run_multiple_em_or_mcem`、`plot_single_iteration` |
| `em.R` | EM 示例：单次 + 500 次取平均 |
| `mcem.R` | MCEM 示例：采样 10/100，单次与多次 |
| `driver_compare500.R` | 500 次平均轨迹图 `41/42` + 配对 EM vs MCEM（$M\in\{20,50,100,200\}$）500 次对比（固定种子 2025，图 42 取 $M=200$）|
| `driver_library_compare.R` | 手写 EM/MCEM 与 `lme4`/`nlme::lme` 在 ML、REML 双口径下逐位对照，输出 `library_compare_summary.txt` |
| `lmm_derivation.md` | EM/MCEM 完整推导 |
| `figures/` | 生成的 PNG 图 |

## 默认参数

`m=50` 组，$n_i\in\{15,20,25\}$，$\beta=(1,2,-1,0.5)$（$p=4$），$G_0=\begin{psmallmatrix}4&1\\1&2\end{psmallmatrix}$（$k=2$），$\sigma^2=1$；
固定初值 $\beta_0=0$，$G_{0,0}=I_2$，$\sigma^2_0=0.5$。

## 复现实验

```bash
cd linear-mixed-model
Rscript driver_compare500.R          # 图 41/42 + 500 次 EM/MCEM 数值对比
Rscript driver_library_compare.R     # ML/REML 并列对照 lme4 与 nlme
# 或交互运行 em.R / mcem.R
```

> 已提交图使用英文标签与 R 基础字体。若自行改为中文标签，可按需用 `showtext` 注册本机 CJK 字体。

## 验证结论（seed=2025）

- **单次 EM**：10 步收敛，观测数据对数似然单调递增；$\hat\beta$、$\hat G_0$ 对角、$\hat\sigma^2$ 逼近真值。
- **500 次模拟取平均**：$\hat\beta\approx(1.00,1.98,-1.00,0.50)$；$\hat G_0$ 对角 $\approx(3.92,1.95)$（非对角在单个数据集上有噪声，多次平均后恢复到 $\approx1$，真值 $1$）；$\hat\sigma^2\approx0.995$；总 MSE $\approx0.13$——与下方 §6.3 主表一致。
- $\hat G_0$ 对角略偏小为方差分量 ML 的固有向下偏；非对角在单个数据集上方差较大，多次平均后准确恢复。
- 若 ML 方差估计表现不理想，应并列报告 REML。`driver_library_compare.R` 已同时验证 `options(lmm_reml=FALSE)` 与 `options(lmm_reml=TRUE)`，并分别对照相同似然口径下的 `lme4`/`nlme`。

### EM vs MCEM，500 次模拟（seed=2025，配对）

所有配置都在**同一批** 500 个数据集上运行。下表为 headline 列；完整的 $M\in\{20,50,100,200\}$ 扫描见顶层 README §6.3。

| 指标 | EM | MCEM(M=200) |
|---|---|---|
| $\hat\beta$ | (1.000, 1.984, −1.000, 0.504) | (1.000, 1.984, −1.000, 0.504) |
| $\hat G_0$ 对角 | (3.921, 1.953) | (3.921, 1.952) |
| $\hat\sigma^2$ | 0.9950 | 0.9950 |
| 总 MSE | 0.13292 | 0.13305 |
| 平均迭代 | 8.34 | 50（达上限）|
| 总耗时 | 30.6 s | 2839 s（≈93×）|

真值 $\beta=(1,2,-1,0.5)$，$G_0$ 对角 $=(4,2)$，$\sigma^2=1$。在同一批数据上，$\hat\beta$ **完全相同**（闭式 GLS），各 $M$ 下 MCEM 都复现 EM（方差分量的 Monte Carlo 噪声随 $M$ 减小）；MCEM 显著更慢且无精度收益，因为 E 步本就有闭式解。
