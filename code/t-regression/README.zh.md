> [English](README.md) | **中文**

# 学生 t 回归模型的 EM / MCEM 估计（论文第 3 章）

## 模型

$$
y_i = x_i^{T}\beta + \varepsilon_i,\qquad \varepsilon_i \sim t_\nu(0,\sigma^2)\ \text{(尺度 } t \text{ 分布)} .
$$

引入隐尺度变量 $z_i\sim\mathrm{Gamma}(\nu/2,\nu/2)$，使 $\varepsilon_i\mid z_i\sim\mathcal N(0,\sigma^2/z_i)$，则边际即为自由度 $\nu$、尺度 $\sigma^2$ 的 $t$ 分布。EM 以 $z=(z_1,\dots,z_n)$ 为隐变量。

**先验**：$\pi(\beta,\sigma^2)\propto 1/\sigma^2$（论文式 3-7），故 $\sigma^2$ 的 M 步除数为 $n+2$。

## 迭代公式

- **E 步权重**：$w_i=\mathbb E[z_i\mid y_i]=\dfrac{\nu+1}{\nu+r_i^2/\sigma^2}$，$r_i=y_i-x_i^{T}\beta$。
- **M 步 $\beta$（加权最小二乘）**：$\hat\beta=(X^{T}WX)^{-1}X^{T}Wy$，$W=\mathrm{diag}(w_i)$。
- **M 步 $\sigma^2$**：$\hat\sigma^2=\dfrac{1}{n+2}\sum_i w_i r_i^2$（对应 $1/\sigma^2$ 先验下的 MAP）。
- **MCEM**：从后验 $z_i\mid y_i\sim\mathrm{Gamma}\!\big(\tfrac{\nu+1}{2},\,\tfrac{\nu+r_i^2/\sigma^2}{2}\big)$ 抽 $M$ 个样本，用样本均值近似 $w_i$。

## 文件

| 文件 | 说明 |
|---|---|
| `data_generation.R` | 生成 $t$ 回归模拟数据；`random_init`（保守）/`bold_random_init`（激进）初始化 |
| `utils.R` | EM/MCEM 迭代（`run_single_em_or_mcem`）、多次模拟（`run_multiple_em_or_mcem`）、MSE、绘图、自由度影响研究 `study_df_effect` |
| `em.R` | EM 示例：单次/多次、初值敏感性、自由度匹配性研究 |
| `mcem.R` | MCEM 示例：采样次数 1/100 的对比 |
| `driver_compare500.R` | EM 与 MCEM(M=100) 各 500 次模拟的可对比性测试（固定种子 2025），出图 `21–24` |
| `driver_em_plots.R` | 单次 EM、保守/激进初始化、自由度扫描图（`25–30`） |
| `driver_mcem_plots.R` | `M=1` 与 `M=100` 的单次 MCEM 迭代路径图（`31–32`） |
| `driver_library_compare.R` | 手写 EM/MCEM 与 `optim`/`hett::tlm`/`MASS::rlm` 逐位对照，输出 `library_compare_summary.txt` |
| `figures/` | 生成的 PNG 图 |

## 默认参数

`n=500, p=2, ν=10, σ²=0.4, β=(2,3,5)`，固定初值 `β₀=(0,0,0), σ²₀=0.1`。

## 复现实验

```bash
cd t-regression
Rscript driver_compare500.R      # 对比图 21–24 + 500 次模拟均值/MSE/耗时
Rscript driver_em_plots.R        # 图 25–30：单次 EM、初始化策略、自由度扫描
Rscript driver_mcem_plots.R      # 图 31–32：单次 MCEM（M=1, M=100）
# 或交互运行 em.R / mcem.R 查看单次与多次迭代图
```

> 已提交图使用英文标签与 R 基础字体。若自行改为中文标签且出现方框，可在脚本前用 `showtext` 注册本机 CJK 字体：
> `library(showtext); font_add("SimSun","<路径>/SimSun.ttc"); showtext_auto(); showtext_opts(dpi=150)`。

## 验证结论

- EM/MCEM 均收敛到先验 $\pi(\beta,\sigma^2)\propto1/\sigma^2$ 下的观测数据 $t$ 对数后验极大值：$\hat\beta\approx(2.02,2.94,4.98)$，$\hat\sigma^2_{\text{MAP}}\approx0.42$。固定效应与独立 `optim` 边际 $t$ MLE 一致；纯 MLE 方差为 $\hat\sigma^2_{\text{MLE}}\approx0.423$，$(n+2)$ 与 $n$ 的除数差异在 $n=500$ 时可忽略，比值 $n/(n+2)\approx0.996$。
- 自由度匹配研究：数据自由度与算法设定一致时 $\sigma^2$ 估计精度最高；收敛速度随自由度增大而提高。

> 完整推导细节见论文第 3 章。
