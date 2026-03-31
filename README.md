## modelscope-guide

本仓库用于指导如何使用 **ModelScope（魔搭）** 进行模型训练、推理调用，以及后续的 **自定义模型** 与 **源码解读**。

当前内容以“先跑通、再理解”为原则，提供可复现的最小示例与工程化目录约定。

## 快速开始（示例跳链）

- 文本分类（StructBERT 微调）：[`bert_text_classification/README.md`](bert_text_classification/README.md)
  - 训练脚本：[`bert_text_classification/train.py`](bert_text_classification/train.py)
  - 推理脚本：[`bert_text_classification/inference.py`](bert_text_classification/inference.py)
  - Notebook：[`bert_text_classification/train.ipynb`](bert_text_classification/train.ipynb)

## 仓库结构

- `bert_text_classification/`：文本分类示例（训练、推理、Notebook、配置）
- `requirements.txt`：本地/个人服务器安装依赖（固定版本，便于复现）
- `requirements/modelscope.txt`：ModelScope 平台环境的兼容依赖（精简）
- `tmp/`：训练与导出产物目录（运行后生成；不建议提交到 Git）

## Roadmap（后续逐步完善）

- 自定义模型：如何注册/组织自定义 `Model`、`Preprocessor`、`Trainer`
- 源码解读：从 `build_trainer`、`Trainer.train()` 到 `pipeline` 的链路梳理
