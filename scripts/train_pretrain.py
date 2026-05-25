#!/usr/bin/env python3
"""
第 2 步（阶段 1）：领域文本继续预训练 / 激活新词嵌入。

- 模型 C（MedVocab）：加 --save_embeddings，训练 embed_tokens + lm_head + LoRA
- 模型 B（DAPT 基线）：同一脚本但不加 --save_embeddings，仅 LoRA

用法示例:
  # 模型 C - 激活新词
  python scripts/train_pretrain.py \\
    --model_path ./qwen2.5-1.5B-MedVocab \\
    --train_file data/sampled_pretrain.jsonl \\
    --output_dir outputs/model_c/stage1_pretrain \\
    --save_embeddings

  # 模型 B - 原始词表 DAPT（需自备原始 Qwen 路径）
  python scripts/train_pretrain.py \\
    --model_path ./models/qwen2.5-1.5B-Instruct \\
    --train_file data/sampled_pretrain.jsonl \\
    --output_dir outputs/model_b/stage1_dapt
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import (
    build_lora_model,
    data_collator,
    jsonl_to_dataset,
    load_base_model,
    load_tokenizer,
    print_trainable_parameters,
    tokenize_pretrain_row,
)


def parse_args():
    p = argparse.ArgumentParser(description="领域 pretrain / 新词激活")
    p.add_argument("--model_path", type=str, default="./qwen2.5-1.5B-MedVocab")
    p.add_argument("--train_file", type=str, default="data/sampled_pretrain.jsonl")
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="outputs/model_c/stage1_pretrain")
    p.add_argument("--save_embeddings", action="store_true", help="MedVocab 必开")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--use_4bit", action="store_true", help="显存不足时开启")
    p.add_argument("--max_steps", type=int, default=-1, help="调试可设 20")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    print("=== 阶段 1: Pretrain / 激活 ===")
    print(f"model_path       : {args.model_path}")
    print(f"save_embeddings  : {args.save_embeddings}")
    print(f"train_file       : {args.train_file}")
    print(f"output_dir       : {args.output_dir}")

    tokenizer = load_tokenizer(args.model_path)
    model = load_base_model(args.model_path, use_4bit=args.use_4bit)
    model = build_lora_model(
        model,
        save_embeddings=args.save_embeddings,
        lora_r=args.lora_r,
    )
    print_trainable_parameters(model)

    dataset = jsonl_to_dataset(args.train_file)
    print(f"训练样本数: {len(dataset)}")

    tokenize_fn = partial(
        tokenize_pretrain_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length
    )
    dataset = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="tokenize pretrain",
    )

    eval_dataset = None
    if args.valid_file:
        eval_dataset = jsonl_to_dataset(args.valid_file).map(
            tokenize_fn,
            remove_columns=jsonl_to_dataset(args.valid_file).column_names,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no" if eval_dataset is None else "steps",
        eval_steps=args.save_steps if eval_dataset else None,
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    collator = partial(data_collator, tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n阶段 1 完成，权重保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
