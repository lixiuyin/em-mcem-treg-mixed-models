> [English](README.md) | **中文**

# EM / MCEM 算法在回归模型参数估计中的应用 — 代码总览

本目录为毕业论文《EM 算法、MCEM 算法的原理及其在回归模型参数估计中的应用》的全部实验代码。
统一用 **EM** 与 **MCEM**（Monte Carlo EM）算法，在三类模型上估计参数并比较性能。

## 目录结构

| 子目录 | 模型 | 对应论文 | 隐变量 |
|---|---|---|---|
| `t-regression/` | 学生 $t$ 回归 $y=x^{T}\beta+\varepsilon$，$\varepsilon\sim t_\nu(0,\sigma^2)$ | 第 3 章 | 尺度变量 $z_i$ |
| `two-level-model/` | 两层线性模型 $y_j=X_jW_j\gamma+X_j\mu_j+\varepsilon_j$ | 第 4 章 | 随机效应 $\mu_j$ |
| `linear-mixed-model/` | 一般 LMM $y_i=X_i\beta+Z_i u_i+\varepsilon_i$（两层模型的推广）| 第 4 章拓展 | 随机效应 $u_i$ |

每个子目录的结构一致：

```
data_generation.R          # 模拟数据生成 + 随机初始化（保守型/扩展型）
utils.R                    # E 步/M 步、单次与多次模拟、边际对数似然、MSE、绘图
                             （two-level-model/ 拆分为 helpers/em_updates/mcem_updates/simulation/plotting/metrics）
em.R / mcem.R              # 示例脚本
driver_*.R                 # 复现实验 + 出图（固定随机种子 2025）
driver_library_compare.R   # 手写 EM/MCEM 与标准库逐位对照（库验证 + 偏差/MSE 评估）
README.md                  # 该实验说明
figures/                   # 生成的 PNG
library_compare_summary.txt  # （位于模块根目录）调库对照输出
```

## 公共约定

- **先验**：第 4 章两层模型与一般 LMM 采用平坦无信息先验 $\pi\propto 1$；第 3 章 $t$ 回归采用 $\pi(\beta,\sigma^2)\propto 1/\sigma^2$。
- **方差分量 M 步除数**：两层模型 $D$ 除以组数 $J$、一般 LMM $G_0$ 除以组数 $m$（MCEM 为 $M\times$组数），$\sigma^2$ 除以总样本量；$t$ 回归 $\sigma^2$ 除以 $n+2$。对混合模型，ML 与 REML 是两种不同估计口径；REML 在方差更新中加入固定效应自由度校正。
- **固定效应**（$\gamma$/$\beta$）：用闭式 GLS（= 给定方差分量时的边际极大似然，ECME 加速）。
- **`get_loglik`**：混合模型 ML 下计算**观测数据边际对数似然**，启用 REML 选项时计算**限制对数似然**；$t$ 回归（先验 $\pi\propto1/\sigma^2$）则计算**观测数据对数后验**（含 $1/\sigma^2$ 先验项）。三者均为对应 EM 口径下单调不减的目标量。
- **MCEM**：固定效应仍解析求解；Monte Carlo 仅用于方差分量更新。

## 运行

```bash
# 任选其一
cd t-regression && Rscript driver_compare500.R    # 及 driver_em_plots.R / driver_mcem_plots.R
cd two-level-model && Rscript driver_em_plots.R   # 及 driver_mcem_plots.R / driver_compare500.R
cd linear-mixed-model && Rscript driver_compare500.R

# 调库对照（手写实现 vs 标准库的逐位验证 + 偏差/MSE 评估）
cd t-regression && Rscript driver_library_compare.R          # vs optim / hett::tlm / MASS::rlm
cd two-level-model && Rscript driver_library_compare.R        # vs lme4（ML 与 REML 两种口径）
cd linear-mixed-model && Rscript driver_library_compare.R     # vs lme4/nlme（ML 与 REML）
```

依赖 R 包：`this.path`、`MASS`、`Matrix`；绘图驱动还会用到 `RColorBrewer`、`viridis`、`ggplot2`、`tidyr`、`dplyr`、`gridExtra`、`scatterplot3d`。调库对照另需 `lme4`、`nlme`，以及可选的 `hett`（缺失时仅跳过 t 回归中的该库对照）。

**字体**：已提交绘图脚本使用英文标签与 R 基础字体。若自行改为中文标签且出现方框，可先注册本机 CJK 字体：

```r
library(showtext); library(sysfonts)
font_add("SimSun", "<路径>/SimSun.ttc"); showtext_auto(); showtext_opts(dpi = 150)
source("driver_xxx.R", encoding = "UTF-8", chdir = TRUE)
```

## 结论速览（seed=2025）

| 模型 | 固定效应 | 方差分量 | 备注 |
|---|---|---|---|
| $t$ 回归 | $\hat\beta\approx(2.02,2.94,4.98)$ | $\hat\sigma^2_{\text{MAP}}\approx0.42$ | 固定效应与边际 $t$-MLE 一致；方差采用 MAP 除数 $n+2$ |
| 两层模型 | $\hat\gamma\approx$ 真值 | $\hat D$ 对角：ML 偏小、REML 校正后无偏 | EM≈MCEM；对数似然单调 |
| 一般 LMM | $\hat\beta\approx$ 真值 | $\hat G_0$：ML 略向下偏，REML 并列报告 | EM≈MCEM；对数似然单调 |

### 调库对照（库验证小结，seed=2025）

`driver_library_compare.R` 在同一批模拟数据上把手写 EM/MCEM 与标准库逐位对照，量化结论：

| 模型 | 对照库 | 单数据集逐位差 | 200/500 次平均 max-diff |
|---|---|---|---|
| $t$ 回归 | `optim`（同后验）、`hett::tlm`、`MASS::rlm` | $\beta$ 差 $2\times10^{-8}$（同后验）；与 `hett`/`rlm` 相关 $1.0$ | $\beta$ 差 $7\times10^{-8}$ |
| 两层模型 | `lme4`（ML / REML）| ML：$\max|\Delta\gamma|=10^{-9}$、对数似然差 $10^{-12}$ | ML $3\times10^{-6}$；REML $3\times10^{-4}$ |
| 一般 LMM | `lme4` / `nlme::lme`（ML / REML）| ML 与 REML 均和对应库口径在方差分量上约 $10^{-7}$ 量级一致 | ML 与 REML 并列报告 |

结论：三套手写实现均与生态金标准在机器精度内吻合，确证了推导与编码的正确性。$t$ 回归 $\hat\sigma^2$ 较纯 MLE 低 $\approx2/n$，源于先验 $\pi\propto1/\sigma^2$（除 $n+2$）而非纯似然（除 $n$），属口径差异而非错误。方差分量在小组数下的 ML 向下偏可由 REML 校正（两层模型 `options(hlm_reml=TRUE)`；一般 LMM `options(lmm_reml=TRUE)`），校正后仍与对应 `lme4`/`nlme` REML 口径逐位一致。

> 论文第 4 章正文相对代码需要的公式/文字/数值口径，见根目录 `README.zh.md` 第 4 章、
> `two-level-model/README.zh.md`，以及 `linear-mixed-model/lmm_derivation.zh.md`。
