# DynaFlow 复现项目（中文）

这是一个面向动态图边异常检测的 DynaFlow 思路复现实现（研究用途，非官方逐行代码）。

## 文档导航
- 主 README（简版）：[README.md](./README.md)
- English version: [README_en.md](./README_en.md)

## 项目能力
- 支持 6 个公开动态图数据集的下载与预处理
- 支持按时间切分训练/测试，并进行可控异常注入
- 包含 DynaFlow 风格流程：
  - 频域增强
  - 图结构聚合
  - 时序建模并输出边异常分数
- 支持以 JSON 导出 AUC/AP/F1 等指标及运行元信息

## 目录结构
```text
dynaflow_scratch/
  scripts/
    download_datasets.sh
  src/dynaflow/
    preprocess.py
    data.py
    model.py
    train.py
  data/
    raw/
    processed/
  results/
```

## 快速开始

### 1) 安装依赖
```bash
cd dynaflow_scratch
pip install -r requirements.txt
```

### 2) 下载原始数据
```bash
./scripts/download_datasets.sh
```

### 3) 预处理为统一 CSV
```bash
PYTHONPATH=src python3 -m dynaflow.preprocess --raw-dir data/raw --out-dir data/processed
```

### 4) 开始训练（最简）
```bash
PYTHONPATH=src python3 -m dynaflow.train --dataset digg --anomaly-ratio 0.10
```

## 常用训练示例

### 模块方式运行（推荐）
```bash
PYTHONPATH=src python3 -m dynaflow.train \
  --dataset uci_messages \
  --anomaly-ratio 0.05
```

### 脚本方式运行
```bash
cd src/dynaflow
python train.py --dataset digg --anomaly-ratio 0.10
```

### 使用样本缓存加速复现实验
```bash
PYTHONPATH=src python3 -m dynaflow.train \
  --dataset digg \
  --anomaly-ratio 0.10 \
  --samples-cache data/processed/cache/digg_r10_cache.npz
```

## 关键参数
- `--dataset`：`uci_messages | digg | email_dnc | bitcoin_alpha | bitcoin_otc | topology`
- `--anomaly-ratio`：测试集异常比例（如 `0.01`、`0.05`、`0.10`）
- 负采样固定使用内置的 StrGNN-style context-dependent sampler。
- `--no-use-spectral`：消融频域增强（w/o spectral）
- `--no-use-low-pass`：保留谱分解但移除 `exp(-beta*lambda)` 低通滤波（w/o low-pass）
- `--temporal-cell`：`garu | gru | none`，用于 GARU vs GRU 与 w/o GARU
- `--hop`：子图阶数；`--hop 0` 可作为 w/o subgraph 设置
- `--window`、`--beta`：时间窗口与低通强度，可用于敏感性分析
- `--samples-cache`：保存/加载采样后的 train/test 边
- `--batch-size`、`--epochs`、`--patience`：核心训练控制参数

## 消融实验

已提供统一脚本补充以下实验：`w/o spectral`、`w/o low-pass`、`w/o GARU`、`GARU vs GRU`、`w/o subgraph`，以及不同 `h`/`w`/`beta` 扫描。

```bash
cd dynaflow_scratch
PYTHONPATH=src python3 scripts/run_ablation.py \
  --datasets uci_messages digg email_dnc bitcoin_alpha bitcoin_otc topology \
  --anomaly-ratio 0.05 \
  --epochs 50 \
  --batch-size 32
```

快速试跑可加样本上限：

```bash
PYTHONPATH=src python3 scripts/run_ablation.py \
  --datasets bitcoin_alpha \
  --epochs 1 \
  --build-train-samples 32 \
  --build-test-samples 32 \
  --max-train-samples 32 \
  --max-test-samples 32 \
  --no-progress \
  --cpu
```

输出位于 `results/ablations/`，包含每个实验的 JSON、checkpoint、`ablation_summary.json` 和 `ablation_summary.csv`。

## 输出结果
- 默认输出路径：`results/{dataset}_rXX.json`
- 指标字段包括：
  - `auc`、`ap`
  - `precision`、`f1`
  - `tpr`、`tnr`
  - `threshold`
  - 样本数量（`n_train`、`n_val`、`n_test`）

## 常见问题
- 直接运行脚本时报导入错误：
  - 优先使用模块方式：`PYTHONPATH=src python3 -m dynaflow.train ...`
- 首次训练在 epoch 前耗时较久：
  - 通常在做样本构建与异常注入
  - 可使用 `--samples-cache ...` 提升后续运行速度
- 没有进度条显示：
  - 检查是否启用了 `--no-progress`

## 免责声明
- 本仓库用于研究复现与实验探索。
- 结果可能与论文官方实现存在差异。
