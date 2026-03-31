> 仓库地址：https://github.com/jinzcdev/modelscope-guide

## 目标读者与学习目标

本 Notebook 面向有后端/前端研发经验、但未系统学习深度学习的同学：你不需要掌握深度学习理论，只要能把训练流程跑通、产出一个可用的文本分类模型，并能在代码里做推理调用即可。

你将学会：

- **在 ModelScope 上拉取数据集**（`MsDataset`）并喂给训练器
- **用预训练 backbone 微调**（`damo/nlp_structbert_backbone_base_std`）
- **理解最小必要的工程要素**：配置文件、work_dir、checkpoint、模型导出目录
- **跑通推理**：用 `pipeline("text-classification")` 对文本列表打分

---

## 平台介绍：Hugging Face vs ModelScope（魔搭）

- **huggingface.co**：生态最广、教程多、模型/数据最多，主流方案以 `transformers + datasets` 为主。
- **modelscope.cn（魔搭）**：对中文任务/中文数据集更友好；提供统一的 `Trainer / Pipeline` 工程化封装；同时支持离线快照与平台开发环境。

本示例选择 ModelScope 的原因：

- **一套 API 跑通训练与推理**：训练用 `build_trainer`，推理用 `pipeline`
- **数据集开箱即用**：`MsDataset.load("DAMO_NLP/yf_dianping")` 直接可用

---

## 示例介绍

- **任务**：文本二分类（正/负面评价）
- **数据集**：`DAMO_NLP/yf_dianping`（大众点评评论）
- **模型 backbone**：`damo/nlp_structbert_backbone_base_std`（StructBERT）
- **训练入口**：`bert_text_classification/train.py`
- **推理入口**：`bert_text_classification/inference.py`

输出目录约定（非常重要）：

- **训练 work_dir**：`tmp/structbert_text_classification`
- **推理模型目录**：`tmp/structbert_text_classification/output`

其中 `output/` 是训练器导出的可用于 `pipeline` 的模型目录（含 `configuration.json`、权重、词表等）。

---

## 环境准备

### 方案 A：个人服务器

最低要求（经验值）：

- **OS**：Linux（Ubuntu 常见）
- **GPU**：>= 8GB 显存（能跑起来的底线；如果是更小的显存，把 `batch_size` 调小也可以，但是训练时间会更长）
- **Python**：建议 3.11+

创建虚拟环境（两种任选其一）：

1. miniconda：

> 参考 [Miniconda 安装指南](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

```bash
conda create -n bert-demo python=3.11 -y
conda activate bert-demo
```

2. venv：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

> 说明：仓库根目录的 `requirements.txt` 为完整的依赖版本。

### 方案 B：ModelScope 平台集成服务器（推荐）

- 可直接使用魔搭平台的免费开发环境（一般自带 CUDA/驱动 与 modelscope 的 Python 环境）
- 使用 `pip install -r requirements/modelscope.txt` 安装兼容的依赖版本即可

---

## 快速开始：一键跑通训练

### 单卡训练

在仓库根目录执行：

```bash
bash bert_text_classification/run_train_single_gpu.sh
```

### 多卡训练（DDP）

示例：用 2 张卡（0,1）：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash bert_text_classification/run_train_multi_gpu.sh
```

---

## 配置文件说明（你只需要理解这几点）

训练配置在 `bert_text_classification/configuration.json`，关键字段：

- **`preprocessor.first_sequence` / `preprocessor.label`**：数据集中作为文本/标签的列名（本示例为 `sentence` / `label`）
- **`train.dataloader.batch_size_per_gpu`**：显存不够就先把它调小（例如 2 → 1）
- **`train.max_epochs`**：先跑通可以设小一点（例如 1）
- **`train.work_dir` 与脚本 work_dir**：真正输出位置由 `train.py` 里的 `WORK_DIR` 控制（见下节）

---

## 训练脚本结构（面向工程的最小理解）

`bert_text_classification/train.py` 做了这些事：

- 从 ModelScope 拉取数据集：
  - `MsDataset.load("DAMO_NLP/yf_dianping", split="train")`
  - `MsDataset.load("DAMO_NLP/yf_dianping", split="validation")`
- 构建训练器并开训（`nlp_base_trainer`）：
  - 预训练 backbone：`damo/nlp_structbert_backbone_base_std`
  - 配置文件：`bert_text_classification/configuration.json`
  - 输出目录：`tmp/structbert_text_classification`
- 根据训练集大小动态修正 `LinearLR.total_iters`（避免写死步数）

---

## 训练产物在哪里？如何用于推理？

训练完成后，你需要关注这两个目录：

- `tmp/structbert_text_classification/`：训练日志与中间产物
- `tmp/structbert_text_classification/output/`：推理需要的“导出模型目录”

推理脚本 `bert_text_classification/inference.py` 直接读取：

- `MODEL_DIR = tmp/structbert_text_classification/output`
- 用 `pipeline("text-classification", model=MODEL_DIR, ...)` 进行推理

---

## 推理快速开始

训练完成后，在仓库根目录执行：

```bash
PYTHONPATH=. python bert_text_classification/inference.py
```

---

## 常见问题（排障优先级从高到低）

- **显存不足（CUDA OOM）**
  - 先把 `configuration.json` 的 `train.dataloader.batch_size_per_gpu` 改小（2 → 1）
  - 也可以把 `evaluation.dataloader.batch_size_per_gpu` 调小
- **首次拉取数据/模型很慢**
  - 属于正常现象：会下载数据集与模型快照；可更换网络环境或提前下载缓存
- **找不到 GPU / device 报错**
  - 先用 `nvidia-smi` 确认驱动与 CUDA 可用
  - `inference.py` 里 `device="cuda"` 可临时改为 `"cpu"`（仅用于验证流程）
- **`pipeline` 找不到模型文件**
  - 确认训练已完成，且 `tmp/structbert_text_classification/output/` 存在
  - 不要把 `MODEL_DIR` 指到 `tmp/structbert_text_classification/` 根目录，推理需要的是 `output/`

---

## Jupyter 示例

可直接打开并逐格运行：

- `bert_text_classification/train.ipynb`

它会演示：

- 安装依赖与环境检查
- 拉取数据集并训练
- 从导出目录加载模型做推理

## 相关工具推荐

1. [ohmyzsh 插件](https://github.com/jinzcdev/ohmyzsh-with-plugins)，让终端更好用

```bash
apt install zsh -y
chsh -s $(which zsh)
sh -c "$(curl -fsSL https://gitee.com/jinzcdev/ohmyzsh-with-plugins/raw/main/install_ohmyzsh.sh)"
```
