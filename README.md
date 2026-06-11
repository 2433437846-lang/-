# 深度学习期末项目：LSTM 与 Transformer/BERT 融合的 RTX 4060 加速序列系统

## 项目概述

这是一个大学深度学习课程的期末项目，核心任务是**电影剧情简介的情感二分类**（正面 / 负面）。项目对比了两种神经网络架构——自定义 **Bi-LSTM** 和预训练 **BERT**——在该任务上的表现，同时进行了 GPU 性能基准测试（FP32 vs AMP 混合精度）和超参数搜索实验。硬件目标为 NVIDIA RTX 4060 Laptop GPU（8 GB 显存）。

---

## 代码架构

项目包含 7 个 Python 模块，按 6 阶段流水线运行：

| 文件 | 职责 |
|---|---|
| `config.py` | 全局配置：路径、超参数、设备/随机种子、中文字体 |
| `data_loader.py` | 数据加载、文本清洗、标签构建、自定义分词器、Dataset/DataLoader |
| `model_lstm.py` | Bi-LSTM 模型定义（Embedding → 双向 LSTM → FC 分类头） |
| `model_bert.py` | BERT 模型封装（HuggingFace `BertForSequenceClassification`） |
| `evaluate.py` | 评估与可视化：混淆矩阵、训练曲线、注意力热力图、对比表 |
| `main.py` | 总控脚本，串联完整的 6 阶段流程 |

### 执行流程

1. **数据准备** — 加载、清洗、划分（70/15/15）、类别分布可视化
2. **模型构建** — 构建 Bi-LSTM 和 BERT
3. **训练** — FP32 训练 + AMP 对比实验
4. **超参数实验** — LSTM 和 BERT 各 3 组实验
5. **测试集评估** — 混淆矩阵、训练曲线、注意力热力图、最终对比表
6. **报告生成** — 输出文本实验报告

---

## 模型详情

### Bi-LSTM（378 万参数，约 14.4 MB）

- Embedding（dim=128）+ Dropout
- 2 层双向 LSTM（hidden_dim=256）
- FC 分类头：Dropout → Linear(512,256) → ReLU → Dropout → Linear(256,2)
- 自定义正则分词器，词表大小 10,038，最大序列长度 256

### BERT（1.09 亿参数，约 417.6 MB）

- `bert-base-uncased`，12 层 Transformer，12 个注意力头，hidden_size=768
- HuggingFace BertTokenizer，最大序列长度 128
- Warmup + Cosine Decay 学习率调度（10% warmup）

---

## 数据集

**文件：** `data/imdb_movies.csv`（6.7 MB）

- **10,178 条**电影记录，使用 `overview`（剧情简介）作为文本输入
- `score ≥ 70` → 正面（1），`score < 70` → 负面（0）
- 类别分布：**正面 32.4%** vs **负面 67.6%**（不平衡）
- 平均文本长度 48 词，划分：7,124 训练 / 1,526 验证 / 1,528 测试

---

## 实验结果

### 测试集性能

| 指标 | Bi-LSTM | BERT |
|---|---|---|
| 准确率 | 67.47% | 69.57% |
| 精确率 | 47.73% | 60.71% |
| 召回率 | 4.24% | 17.17% |
| F1 | 7.79% | 26.77% |
| 最佳验证准确率 | 67.82% | 70.45% |

### 训练效率

| 指标 | Bi-LSTM | BERT |
|---|---|---|
| 总训练时间 | 22.2s | 222.9s |
| 每轮时间 | 5.5s | 74.3s |
| 峰值显存 | 264.6 MB | 3,853.1 MB |

BERT 慢约 10 倍，显存占用约 14.5 倍，符合其 29 倍参数量的预期。

### AMP vs FP32

| 模型 | FP32 显存 | AMP 显存 |
|---|---|---|
| Bi-LSTM | 264.6 MB | 1,074.0 MB |
| BERT | 3,853.1 MB | 4,021.2 MB |

AMP 未能节省显存——LSTM 因 GradScaler 开销反而增加，BERT 差异可忽略。

### 超参数实验

**LSTM（3 组）：** 最佳验证准确率均在 67.8–68.6% 附近，较低学习率（5e-4）略优达 68.55%。

**BERT（3 组）：** 验证准确率 70.05%–71.63%，最优配置为 lr=2e-5 + batch_size=16（71.63%）。

---

## 项目产出

所有输出位于 `outputs/` 目录：

- `class_distribution.png` — 训练/验证/测试集类别分布柱状图
- `confusion_matrix_Bi-LSTM.png` / `confusion_matrix_BERT.png` — 混淆矩阵热力图
- `training_comparison.png` — 四面板对比图：损失曲线、准确率曲线、每轮耗时、参数量
- `gpu_profile_table.png` — FP32 vs AMP 对比表
- `lstm_hyperparam_table.png` / `bert_hyperparam_table.png` — 超参数实验结果表
- `final_comparison.png` — Bi-LSTM vs BERT 最终对比表
- `experiment_report.txt` — 文本实验报告

模型检查点保存在 `checkpoints/`：

- `best_lstm_model.pth`（15.1 MB）
- `best_bert_model.pth`（438 MB）

TensorBoard 日志保存在 `runs/` 目录。

---

## 已知问题

1. **类别不平衡严重** — 两个模型实质上退化为多数类预测器，正面召回率极低（LSTM 4.24%、BERT 17.17%），未使用类别加权、过采样或 Focal Loss 等技术
2. **BERT 注意力热力图失败** — `transformers` 版本不兼容 `attn_implementation` 属性
3. **AMP 未节省显存** — LSTM 的 AMP 显存反而增加（GradScaler 开销），BERT 差异可忽略

---

## 依赖环境

```
torch>=2.0
transformers>=4.30
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorboard
tqdm
```

运行环境：PyTorch 2.11.0 + CUDA 12.8 + RTX 4060 Laptop GPU
