# AI4S_ThinFilm_2023270047

**MLP-Based Spectral Prediction and Data-Driven Design of Multilayer Dielectric Thin Films**

薄膜技术 · AI4S 课程大作业 —— 基于传输矩阵法（TMM）与多层感知机（MLP）的多层介质膜光谱预测与辅助设计。

| 个性化参数 | 值 |
| --- | --- |
| 学号 | 2023270047 |
| 学号后两位 `N` | 47 |
| **λtarget** | **610 nm**  （= 450 + 10 × (47 mod 31)） |
| **seed** | **270047**  （学号后 6 位） |
| **design_seed** | **270048**  （= seed + 1） |
| 目标波长对应索引 | 21（λ = 400 + 10 × 21 = 610 nm） |

---

## 1. 研究内容

用传输矩阵法（TMM）生成"膜厚 → 反射光谱"数据，训练一个简单的 MLP 代理模型，
再用它在上万组候选膜系中快速筛选目标波长处反射率最高的设计，最后**用 TMM 重新验证**。

四个必做环节与三个科研问题：

| 环节 | 内容 | 对应脚本/输出 |
| --- | --- | --- |
| ① TMM 与数据生成 | 用 `seed` 生成 5000 组膜厚与光谱，固定划分 4000/500/500 | `main.py` → `data/dataset.npz` |
| ② MLP 光谱预测 | `[d1,d2,d3,d4] → R(λ)`，结构 4-128-128-64-41，MSE 损失 | `main.py` → `data/mlp_full.pth` |
| ③ 数据量影响 | 从同一 4000 组训练池依次取前 500/1000/2000/4000 组，验证集与测试集不变 | `main.py` → `data/training_size_metrics.csv` |
| ④ MLP 辅助设计 | 用 `design_seed` 生成 10000 组新候选，按 λtarget 排序，再用 TMM 验证 | `main.py` → `data/Table1_top5_designs.csv` |

- **Q1** MLP 能否准确预测反射光谱？
- **Q2** 训练数据量增加后，预测误差如何变化？
- **Q3** MLP 能否作为代理模型帮助快速筛选目标膜系？

### 统一研究对象

| 项目 | 设置 |
| --- | --- |
| 膜系结构 | Air / H / L / H / L / Glass，共 4 层 |
| 折射率 | nH = 2.30，nL = 1.45，ns = 1.52，n0 = 1.0（空气） |
| 膜厚范围 | 每层 40–180 nm |
| 波长范围 | 400–800 nm，步长 10 nm，共 41 点（正入射，忽略吸收与色散） |
| 数据集 | 5000 组 = 4000 训练 + 500 验证 + 500 测试（固定划分，全部实验共用） |
| MLP | 输入 4 个膜厚 → 输出 41 点反射光谱，结构 4-128-128-64-41，ReLU |
| 训练 | Adam，lr = 1×10⁻³，batch size 64，主实验 2000 epoch，训练量实验 1000 epoch |
| 设计筛选 | design_seed 生成 10000 组候选 → MLP 排序取前 10 → TMM 验证 → 最终 Top 5 |

---

## 2. 环境配置

Python 3.11。依赖很少，CPU 即可完成全部实验（本工作未使用 GPU）。

```bash
# 方式一：pip
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 方式二：conda
conda env create -f environment.yml
conda activate ai4s-thinfilm
```

`requirements.txt` 给出的是**产出论文数值时实际使用的版本**：

```
torch==2.14.0
numpy==2.4.6
pandas==3.0.6
matplotlib==3.11.2
```

---

## 3. 目录结构

```
.
├── main.py                    # 主程序：数据生成 → 训练 → 评估 → 训练量实验 → MLP 辅助设计
├── make_figures.py            # 依据已保存的权重与结果表重新生成 Figure 7 / Figure 8
├── config.json                # 全部关键参数（λtarget、seed、design_seed、结构、训练超参）
├── requirements.txt
├── environment.yml
├── LICENSE
├── data/
│   ├── dataset.npz            # 5000 组膜厚-光谱 + 4000/500/500 划分 + wavelengths/seed/design_seed/lambda_target
│   ├── mlp_full.pth           # 主实验模型（2000 epoch，4000 组训练）
│   ├── mlp_n500.pth           # 训练量实验：500 组
│   ├── mlp_n1000.pth          #               1000 组
│   ├── mlp_n2000.pth          #               2000 组
│   ├── mlp_n4000.pth          #               4000 组
│   ├── test_predictions.npz   # 测试集预测/真值/输入
│   ├── training_size_metrics.csv   # 表：训练量 → MSE / MAE / RMSE / R²
│   └── Table1_top5_designs.csv     # 论文 Table 1：Top 5 设计的膜厚与 MLP/TMM 反射率
├── figures/
│   ├── Figure4_loss_vs_epoch.png
│   ├── Figure5_test_samples.png
│   ├── Figure6_training_size.png
│   ├── Figure7_design_spectrum.png
│   └── Figure8_failure_case.png
└── report/
    ├── AI4S_Research_Report_2023270047.docx    # 论文（Research Article 模板，6 页）
    └── AI4S_Research_Report_2023270047.pdf
```

> `main.py` / `make_figures.py` 会自动使用 `data/` 与 `figures/` 子目录；若这两个目录
> 不存在，则回退到脚本所在目录，因此把两个 .py 单独拷到别处也能直接运行。

---

## 4. 数据生成

```bash
python main.py
```

`main.py` 是**完整可复现的单文件流程**，依次完成：数据生成 → 训练主模型 → 测试集评估
→ 训练量实验 → MLP 辅助设计 → 输出全部图表与结果表。

- 若 `data/dataset.npz` 不存在，会先用 `seed = 270047` 生成 5000 组数据（约 1–2 分钟，
  5000 × 41 次 2×2 复数矩阵连乘）；
- 若已存在，则直接加载，跳过生成。

数据生成的确定性来自：

```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # seed = 270047
X = np.random.uniform(40.0, 180.0, size=(5000, 4))                  # 膜厚
rng = np.random.RandomState(seed); perm = rng.permutation(5000)      # 固定重排
train = perm[:4000]; val = perm[4000:4500]; test = perm[4500:5000]   # 固定划分
```

`dataset.npz` 中同时保存了 `wavelengths`、`seed`、`design_seed`、`lambda_target`，
可用来自查数据是否与本人参数一致：

```python
import numpy as np
d = np.load('data/dataset.npz')
print(int(d['seed']), int(d['design_seed']), int(d['lambda_target']))   # 270047 270048 610
print(d['X'].shape, d['Y'].shape, d['test_X'].shape)                    # (5000,4) (5000,41) (500,4)
```

---

## 5. 模型训练

训练已包含在 `main.py` 中，无需单独调用：

```bash
python main.py        # 训练主模型（2000 epoch）+ 四个训练量模型（各 1000 epoch）
```

关键实现（`main.py`）：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, NUM_WL)          # NUM_WL = 41
        )
```

- 损失函数 `nn.MSELoss()`，优化器 `torch.optim.Adam(lr=1e-3)`，`batch_size=64`；
- **每个 epoch 结束后在验证集上评估一次，取验证损失最小时的权重作为最终模型**；
- 权重初始化由 `torch.manual_seed(270047)` 控制，因此在 CPU 上完全可复现。

---

## 6. 结果复现

### 6.1 一步复现全部结果

```bash
python main.py
```

运行结束后会在 `data/` 与 `figures/` 下重新写出 `dataset.npz`、`mlp_*.pth`、
`test_predictions.npz`、`training_size_metrics.csv`、`Table1_top5_designs.csv`
以及 `Figure4`–`Figure8`。

### 6.2 论文中的关键数值（主实验，seed = 270047）

测试集 500 组样本 × 41 个波长点：

| 指标 | 数值 |
| --- | --- |
| MSE | 8.906 × 10⁻⁴ |
| MAE | 0.0216 |
| RMSE | 0.0298 |
| R² | 0.9725 |

在 λtarget = 610 nm 处单独统计：R² = 0.9842，MAE = 0.0177，最大绝对误差 = 0.0833。

### 6.3 训练数据量影响（Q2）

从同一 4000 组训练池依次取前 500/1000/2000/4000 组，验证集与测试集始终不变，
统一训练 1000 epoch（对应 `data/training_size_metrics.csv`）：

| 训练样本数 | MSE | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| 500 | 0.0220 | 0.1211 | 0.1482 | 0.3205 |
| 1000 | 0.0133 | 0.0918 | 0.1153 | 0.5885 |
| 2000 | 0.0036 | 0.0451 | 0.0601 | 0.8882 |
| 4000 | 0.0014 | 0.0261 | 0.0372 | 0.9571 |

R² 从 0.321 升到 0.957，RMSE 由 0.1482 降到 0.0372（降幅约 75%）。

> 复现该表需要重新训练 4 个模型；`data/mlp_n*.pth` 已保存，
> 可直接加载比较而不必重训。

### 6.4 MLP 辅助设计与 TMM 验证（Q3，论文 Table 1）

用 `design_seed = 270048` 生成 10000 组新候选（与训练数据不重复）→ MLP 一次前向推理
→ 按 610 nm 处预测反射率降序取前 10 → 对这 10 个候选逐个用 TMM 计算真实光谱
→ 按 610 nm 反射率重新排序，取 Top 5：

| Rank | d1/nm | d2/nm | d3/nm | d4/nm | MLP Rtarget | TMM Rtarget |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 67.19 | 105.61 | 65.89 | 122.66 | 0.6833 | **0.6571** |
| 2 | 66.82 | 111.32 | 68.28 | 90.66 | 0.6764 | 0.6549 |
| 3 | 63.26 | 115.78 | 57.83 | 98.70 | 0.6750 | 0.6523 |
| 4 | 64.97 | 119.42 | 61.86 | 88.87 | 0.6741 | 0.6512 |
| 5 | 73.89 | 101.74 | 59.53 | 81.73 | 0.6680 | 0.6464 |

- 最优设计：TMM 在 610 nm 处反射率 **0.6571**（MLP 预测 0.6833）；
- Top 5 上 MLP 系统性高估，平均绝对偏差 0.0230，平均相对偏差 **3.52%**，最大 3.99%；
- 作为参照，数据集内 610 nm 处反射率均值 0.2944、最大值 0.6537；
- **结论：代理模型适合"粗筛"，最终裁决必须由 TMM 完成。**

### 6.5 失败案例

测试集中全波段平均平方误差最大的样本（索引 273）：
RMSE = 0.1320（测试集样本 RMSE 中位数 0.0256），全波段最大绝对误差 0.2760。
该样本上 MLP 预测的峰值位置与 TMM 一致，但峰值幅度被压缩——误差集中在
反射率变化剧烈的波段，对应网络对高频起伏的平滑化倾向。

### 6.6 一键校验论文全部数值

```bash
python verify_results.py
```

该脚本**只读取 `main.py` 产出的文件**（`dataset.npz`、`mlp_full.pth`、
`training_size_metrics.csv`、`Table1_top5_designs.csv`），独立重算论文里的每一个
数字（测试集 MSE/MAE/RMSE/R²、610 nm 处指标、失败样本、训练量对照表、Top 5 设计的
MLP/TMM 反射率与偏差），并逐条与论文数值对照；同时用一个**独立编写的 TMM 实现**
复核 `dataset.npz` 与 Table 1，确认数值确实来自传输矩阵法而不是伪造。全部一致时
输出 `结果: 全部通过`。

### 6.7 只重画 Figure 7 / Figure 8

```bash
python make_figures.py      # 需要 data/mlp_full.pth 与 data/dataset.npz 已存在
```

---

## 7. 薄膜筛选方法（三步流程）

```
               10000 组随机候选膜厚 (design_seed = 270048)
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │  ① MLP 代理模型粗筛                         │
        │  一次前向推理得到全部候选的 R(λ)             │
        │  按 λtarget = 610 nm 处预测反射率降序排序    │
        └───────────────────────────────────────────┘
                            │  取前 10
                            ▼
        ┌───────────────────────────────────────────┐
        │  ② TMM 物理精算                             │
        │  对这 10 个候选逐个计算真实反射光谱           │
        └───────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │  ③ 按 TMM 的 R(610 nm) 重新排序 → Top 5      │
        └───────────────────────────────────────────┘
```

**为什么要两步？** 对 10000 组候选做完整 TMM 扫描，需要在 41 个波长点上各完成一次
四层 2×2 复数矩阵连乘；而 MLP 只需**一次前向推理**就能给出全部候选的光谱估计。
先粗筛再精算，把昂贵的物理计算留给少量有希望的候选。但 Table 1 显示 MLP 在
Top 5 上系统性偏高约 3.52%，因此**不能用预测值直接下结论，必须由 TMM 复核**。

---

## 8. 重要说明

- 论文正文中 Figure 1–3 为**示意图**（研究框架、TMM 物理模型与数据表示、MLP 结构），
  由绘图脚本绘制，不构成数值结果；
- **Figure 4–8 与 Table 1 全部由本仓库代码产出**，可由 `python main.py` 重新生成；
- 全部数据、随机种子与目标波长均由学号确定：

  ```python
  STUDENT_ID = "2023270047"
  N            = int(STUDENT_ID[-2:])              # 47
  SEED         = int(STUDENT_ID[-6:])              # 270047
  DESIGN_SEED  = SEED + 1                          # 270048
  LAMBDA_TARGET = 450 + 10 * (N % 31)              # 610 nm
  ```
