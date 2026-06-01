> [English](empirical_README.md) | **中文**

# 深度学习真实数据实证分析 —— 复现说明

对应论文**第 5 章**。两条流水线分别把论文第 3、4 章的 EM/MCEM 实现应用于真实数据，**算法代码复用** `t-regression/` 与 `two-level-model/`，仅替换数据。

## 环境

- **R 4.5**：`this.path`、`MASS`、`Matrix`、`lme4`（金标准对照）。重绘脚本使用英文标签、macOS `quartz` PNG 设备与 `CJK <- "sans"`；非 macOS 如有需要请调整 PNG 设备类型。
- **Python 3.13**：`torch`、`torchvision`、`datasets`、`scikit-learn`、`numpy`、`pandas`。嵌入抽取自动使用 MPS（Apple GPU），无则回退 CPU。
- 随机种子统一 `20250529`（Python 抽样/子采样、R 端 MCEM）。

## 已提交数据与可再生成产物

为保持仓库轻量，仅提交体积较小的原始输入，以及 R 分析**直接读取**的 CSV；体积较大的派生/中间文件已被 git 忽略，由第 1 步脚本再生成：

| 状态 | 文件 | 说明 |
|---|---|---|
| 已提交 | `empirical-2level-cifar10h/data/cifar10h-raw.zip` | CIFAR‑10H 原始 trial；`01_build_lmm_data.py` 直接读取（无需解压） |
| 已提交 | `empirical-2level-cifar10h/data/cifar10h-probs.npy` | 人类软标签概率 |
| 已提交 | `empirical-2level-cifar10h/data/cifar10h_lmm_main.csv` | `02_lmm_real.R` / `03_replot.R` 读取的 J=300 主数据集 |
| 已提交 | `empirical-t-utkface/data/utkface_t_regression.csv` | `02_tregression_real.R` / `03_replot.R` 读取的回归数据集 |
| 再生成 | `empirical-2level-cifar10h/data/cifar10h_lmm_full.csv` | `04_robustness_full.R` 的全量稳健性数据集；运行 `01_build_lmm_data.py` 生成 |
| 再生成 | `empirical-t-utkface/data/utkface_embeddings.npz` | ResNet‑50 嵌入；运行 `01_extract_embeddings.py`（需 Hugging Face 下载 + GPU/CPU） |
| 再生成 | `empirical-t-utkface/output/treg_results.rds` | `03_replot.R` 读取的模型拟合结果；运行 `02_tregression_real.R` 生成 |
| 再生成 | `empirical-2level-cifar10h/output/lmm_results.rds` | `03_replot.R` 读取的模型拟合结果；运行 `02_lmm_real.R` 生成 |
| 再生成 | `empirical-2level-cifar10h/output/lmm_full_robustness.rds` | 全量稳健性拟合结果（J=2567）；运行 `04_robustness_full.R` 生成（需 `cifar10h_lmm_full.csv`） |

## 流水线 1：t 回归 —— `empirical-t-utkface/`

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1 | `code/01_extract_embeddings.py` | 从 Hugging Face `py97/UTKFace-Cropped`（MIT，约 107 MB）抽 8000 张人脸，ResNet‑50 提 2048 维嵌入 → PCA 30 维 → `data/utkface_t_regression.csv` |
| 2 | `code/02_tregression_real.R` | OLS 基线；ν 的 profile 边际似然选 ν*；ν* 下 EM/MCEM；与 `optim` 边际 MLE、`MASS::rlm` 双重验证；输出 `treg_summary.txt` 与再生成的 `treg_results.rds` |
| 3 | `code/03_replot.R` | 从已保存结果重绘 4 张图 |

**关键结果**：ν*=6；t 边际似然较正态 +151；EM vs optim 系数最大差 0.0002，EM vs rlm 相关 1.0；被最强下调 10% 样本中年龄≥60 占 59.5%（全样本 11.3%）。

> 注：论文原始 `t-regression/utils.R` 用 `diag(gamma)`（$n\times n$ 稠密阵），仅适用于 $n\approx500$ 的模拟；`02_tregression_real.R` 改用按行缩放 `w*X`（数学等价于式 3-14/3-19/3-22），从而支持 $n=8000$。

## 流水线 2：两层线性模型 —— `empirical-2level-cifar10h/`

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1 | `code/01_build_lmm_data.py` | 读取已提交的 CIFAR‑10H 原始 trial（`cifar10h-raw.zip`，源自 `github.com/jcpeterson/cifar-10h`）+ 软标签概率；清洗反应时；图像难度=人类软标签熵；取 J=300 标注者 → `data/cifar10h_lmm_main.csv`（全量另存为 `_full.csv`，已 git 忽略） |
| 2 | `code/02_lmm_real.R` | 复用论文两层模型 EM/MCEM（**仅覆盖** `get_X_j/y_j/W_j` 三个访问器以读分组数据，避免巨型块对角阵）；剔除 1 个秩亏标注者，拟合实际用 **J=299**（N=59699）；EM + MCEM(M=20/50/200)；`lme4` 金标准对照；输出 `lmm_summary.txt` 与再生成的 `lmm_results.rds` |
| 3 | `code/03_replot.R` | 从已保存结果重绘 3 张图 |
| 4 | `code/04_robustness_full.R` | 稳健性检验：在全量标注者集（剔除秩亏标注者后 J=2567）上重跑确定性 EM，核验固定效应 / σ² / ICC 与 J=299 主分析一致；读取 `cifar10h_lmm_full.csv`（由第 1 步再生成），打印控制台摘要并写出 `output/lmm_full_robustness.rds` |

**关键结果**：γ=(截距 7.65, 难度 +0.13, 正确 −0.26, 试次 −0.03)；σ²=0.1191 与 lme4 完全一致（边际对数似然 −22423.06 逐位吻合），随机效应 SD 吻合到 3 位小数；截距 ICC=0.473；MCEM 与 EM 吻合到 4 位小数。

> 模型设定 q=0、$W_j=I_4$（四个一级系数：截距、难度、正确、试次），即随机系数模型 $\boldsymbol\beta_j\sim N(\boldsymbol\gamma,D)$，对应论文式 (4‑25b) GLS 闭式 γ 更新与式 (4‑32) 平坦先验下除数为 J 的 D 更新。

## 运行顺序

```bash
# 流水线1
python3 empirical-t-utkface/code/01_extract_embeddings.py
Rscript  empirical-t-utkface/code/02_tregression_real.R
Rscript  empirical-t-utkface/code/03_replot.R
# 流水线2
python3 empirical-2level-cifar10h/code/01_build_lmm_data.py
Rscript  empirical-2level-cifar10h/code/02_lmm_real.R
Rscript  empirical-2level-cifar10h/code/03_replot.R
Rscript  empirical-2level-cifar10h/code/04_robustness_full.R   # 可选：全量稳健性检验
```
