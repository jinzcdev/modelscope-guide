#!/usr/bin/env python
"""
任务描述：在 DAMO_NLP/yf_dianping 数据集上微调文本分类（StructBERT / BERT 系 backbone）

要点：
1. model 传 ModelScope Hub 上的预训练 backbone，权重与 tokenizer 从快照目录加载；
2. cfg_file 指向本目录 configuration.json，只描述 task / train / preprocessor 等训练侧配置；
3. cfg_modify_fn 从训练集+验证集统计 label2id，并同步 model.num_labels 与 LinearLR 步数

运行：
  bash bert_text_classification/run_train_single_gpu.sh
  bash bert_text_classification/run_train_multi_gpu.sh
"""

import os

from modelscope import EpochBasedTrainer
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile

# 工作目录，用于保存训练过程中的日志、checkpoint 等
WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(
    __file__), "..", "tmp", "structbert_text_classification"))
# 本目录的 configuration.json：训练超参、preprocessor 类型等
CFG_FILE = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), ModelFile.CONFIGURATION)

# 数据集中第一个字段以及label字段的字段名，用于读取对应列的数据
FIRST_SEQUENCE_KEY = "sentence"
LABEL_KEY = "label"


def build_cfg_modify_fn(train_ds: MsDataset):
    """返回 cfg_modify_fn：写入 LinearLR total_iters。"""

    def cfg_modify_fn(cfg):

        if cfg.train.lr_scheduler.type == "LinearLR":
            bs = cfg.train.dataloader.batch_size_per_gpu
            steps_per_epoch = max(1, int(len(train_ds) / bs))
            cfg.train.lr_scheduler.total_iters = steps_per_epoch * cfg.train.max_epochs

        return cfg

    return cfg_modify_fn


def main():

    # 加载训练集和验证集
    train_dataset = MsDataset.load(
        "DAMO_NLP/yf_dianping", split="train", subset_name="default")
    eval_dataset = MsDataset.load(
        "DAMO_NLP/yf_dianping", split="validation", subset_name="default")

    # 构建 cfg_modify_fn, 用于在运行时修改配置文件，比如这里 LinearLR 的步数是需要根据训练集的大小来计算的
    cfg_modify_fn = build_cfg_modify_fn(train_dataset)

    kwargs = dict(
        # 预训练中文句向量/分类常用 backbone（StructBERT）
        model="damo/nlp_structbert_backbone_base_std",
        cfg_file=CFG_FILE,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        seed=42,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn,
    )
    # torchrun 会设置 WORLD_SIZE>1，需传入 launcher 以注册 DDPHook（与 finetune_text_classification 一致）
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        kwargs["launcher"] = "pytorch"
    # NLP 任务建议使用 nlp_base_trainer：Tokenizer 构建等与官方单测一致
    trainer: EpochBasedTrainer = build_trainer(
        name=Trainers.nlp_base_trainer,
        default_args=kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
