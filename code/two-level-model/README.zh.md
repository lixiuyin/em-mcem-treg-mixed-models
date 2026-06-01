> [English](README.md) | **中文**

# 两层线性模型的 EM / MCEM 估计（论文第 4 章）

## 模型

$$
y_j = X_j\beta_j+\varepsilon_j,\qquad \beta_j=W_j\gamma+\mu_j,\qquad j=1,\dots,J,
$$
代入得 $y_j = X_j W_j\gamma + X_j\mu_j+\varepsilon_j$，其中 $\mu_j\sim\mathcal N(0,D)$、$\varepsilon_j\sim\mathcal N(0,\sigma^2 I_{n_j})$。
待估参数：固定效应 $\gamma$、二级协方差 $D$、个体误差方差 $\sigma^2$。隐变量为 $\mu=(\mu_1,\dots,\mu_J)$。

**先验约定（已统一）**：平坦无信息先验 $\pi(\gamma,D,\sigma^2)\propto 1$ ⇒ $D$ 的 M 步除数为 $J$（MCEM 为 $M\!\cdot\!J$），$\sigma^2$ 除以总样本量 $N$。
此约定下 `get_loglik`（观测数据边际对数似然）正是 EM 单调不减的目标量。

## 迭代公式（与论文式对应）

- **E 步**（式 4-17/4-21）：$\mu_{j}^{*}=(X_j^{T}X_j+\sigma^2 D^{-1})^{-1}X_j^{T}(y_j-X_jW_j\gamma)$，$V_j^{*}=\sigma^2(X_j^{T}X_j+\sigma^2 D^{-1})^{-1}$。
- **$\gamma$ 更新**（闭式 GLS，式 4-25b）：$\hat\gamma=(\sum_j W_j^{T}\hat\Lambda_j^{-1}W_j)^{-1}\sum_j W_j^{T}\hat\Lambda_j^{-1}\hat\beta_j^{\text{OLS}}$，$\hat\Lambda_j=D+\sigma^2(X_j^{T}X_j)^{-1}$。
- **$D$ 更新**（式 4-32，÷$J$）：$\hat D=\frac{1}{J}\sum_j(\mu_j^{*}\mu_j^{*T}+V_j^{*})$。
- **$\sigma^2$ 更新**（式 4-35，÷$N$）：$\hat\sigma^2=\frac1N\sum_j[\lVert y_j-X_jW_j\gamma-X_j\mu_j^{*}\rVert^2+\mathrm{tr}(X_jV_j^{*}X_j^{T})]$。
- **MCEM**：$\gamma$ 仍用闭式 GLS；$D=\frac{1}{MJ}\sum_l\sum_j\mu_j^{(l)}\mu_j^{(l)T}$，$\sigma^2=\frac{1}{MN}\sum_l\sum_j\lVert\cdot\rVert^2$。

## 文件

| 文件 | 说明 |
|---|---|
| `data_generation.R` | 生成两层模型数据（J 组，$n_j\in\{60,80,100\}$）；`random_init`/`bold_random_init` |
| `utils.R` | 入口文件：加载以下所有拆分模块；其他脚本只需 source `utils.R` |
| `helpers.R` | 数据访问器：`get_X_j`、`get_y_j`、`get_W_j`、`get_mu_j`、后验辅助函数 |
| `em_updates.R` | EM E 步/M 步：`em_update_gamma`、`em_update_D`、`em_update_sigma2` |
| `mcem_updates.R` | MCEM 采样与更新：`get_mu_samples`、`mcem_update_all` |
| `simulation.R` | `run_single_em_or_mcem`、`run_multiple_em_or_mcem` |
| `plotting.R` | 所有 `plot_*` 绘图函数：迭代路径、MSE、缩放视图 |
| `metrics.R` | `get_mse`、`get_loglik`、`print_result` |
| `em.R` | EM 示例：单次、500 次取平均、两种初值敏感性、与 MCEM 对比 |
| `mcem.R` | MCEM 示例：采样 10/100，单次与 500 次 |
| `driver_em_plots.R` | 生成图 `01–08`（论文图 4.1/4.2） |
| `driver_mcem_plots.R` | 生成图 `11–18`（论文图 4.3/4.4）|
| `driver_compare500.R` | EM vs MCEM(M=50) 500 次对比，出图 `51–54`（固定种子 2025，REML）|
| `driver_library_compare.R` | 手写 EM/MCEM 与 `lme4`（ML/REML 双口径）逐位对照，输出 `library_compare_summary.txt` |
| `figures/` | 生成的 PNG 图 |

## 默认参数

`J=20, p=2, q=3`，$n_j\in\{60,80,100\}$。真实参数按论文 §4.1.2（在 `set.seed(2025)` 下随机抽取）：$D=Q\,\mathrm{diag}(5,6,7)\,Q^{\top}$，$Q$ 为随机正交矩阵（特征值 $(5,6,7)$），$\gamma_i\sim U(0,12)$，$\sigma^2\sim U(1,5)$（抽得 $\sigma^2\approx2.693$）。
固定初值 $\hat\gamma_0=(0.1,\dots,0.1)$，$\hat D_0=\begin{psmallmatrix}4&2&1\\2&5&3\\1&3&6\end{psmallmatrix}$，$\hat\sigma^2_0=0.1$；容差 $10^{-6}$（绘图驱动用最大迭代 50，`driver_compare500.R` 用 30）。

## 复现实验

```bash
cd two-level-model
Rscript driver_em_plots.R       # 图 01–08
Rscript driver_mcem_plots.R     # 图 11–18（含 M=100×500 次，较慢）
Rscript driver_compare500.R     # 图 51–54 + EM/MCEM 数值对比
```

> 已提交图使用英文标签与 R 基础字体。若自行改为中文标签，可按需用 `showtext` 注册本机 CJK 字体。

## 验证结论（seed=2025，500 次模拟，`max_iter=30`）

方差分量 $D$ 采用 **REML** 估计（`driver_compare500.R` 设 `options(hlm_reml=TRUE)`，见主 README §4.5.2）：

- $\hat\gamma$、$\hat\sigma^2$ 准确恢复真值；总 MSE（对 500 次估计先平均再算偏差范数）：EM $\approx0.453$，MCEM(M=50) $\approx0.368$（同一量级）。
- $\hat D$（REML）能较好恢复真实 $D$（Frobenius MSE $\approx0.28$），在 $J=20$ 的小组数下即基本无偏。若改用 ML（`options(hlm_reml=FALSE)`）则 $\hat D$ 整体缩为约 $(J-q-1)/J\approx0.8$ 倍（各特征值约乘 $0.8$，即低估约 20%）——这是方差分量极大似然的固有向下偏（非 bug，`lme4` ML 给出相同结果），REML 通过传播 $\hat\gamma$ 的不确定性消除该偏。
- 观测/限制对数似然单调递增；EM(ML)/EM(REML) 分别与 `lme4`(ML)/`lme4`(REML) 逐位一致（详见主 README §6.2.1 与 `driver_library_compare.R`）。

> 与论文正文（第 4 章 / §6.2）相关内容已与本代码核对一致：平坦先验（$\pi\propto1$，故 $D$ 除以 $J$）、QR-随机真值设定、以及更新后的表 4-2 数值。完整推导与结果见根目录 `README.zh.md`（§4、§6.2）。
