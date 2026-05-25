#!/usr/bin/env python3
"""
第 2 步冒烟：不真正训练，只验证数据 + 模型 + 一条前向能否跑通。
用法:
  python scripts/verify_train_setup.py --stage pretrain
  python scripts/verify_train_setup.py --stage sft --save_embeddings
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch

# 允许从 scripts/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_common import (
    build_lora_model,
    data_collator,
    jsonl_to_dataset,
    load_base_model,
    load_tokenizer,
    tokenize_pretrain_row,
    tokenize_sft_row,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    p.add_argument("--model_path", type=str, default="./qwen2.5-1.5B-MedVocab")
    p.add_argument("--save_embeddings", action="store_true")
    p.add_argument("--max_seq_length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    data = root / "data"

    if args.stage == "pretrain":
        train_file = data / "sampled_pretrain.jsonl"
        if not train_file.is_file():
            print(f"缺少 {train_file}，请先运行 scripts/sample_data.py")
            sys.exit(1)
    else:
        train_file = data / "sampled_sft_train.jsonl"
        if not train_file.is_file():
            print(f"缺少 {train_file}，请先运行 scripts/sample_data.py")
            sys.exit(1)

    print(f"stage={args.stage}, model={args.model_path}")
    tokenizer = load_tokenizer(args.model_path)
    model = load_base_model(args.model_path, use_4bit=False)
    model = build_lora_model(model, save_embeddings=args.save_embeddings)

    ds = jsonl_to_dataset(train_file).select(range(4))
    if args.stage == "pretrain":
        fn = partial(tokenize_pretrain_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    else:
        fn = partial(tokenize_sft_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    ds = ds.map(fn, remove_columns=ds.column_names)

    batch = data_collator([ds[i] for i in range(len(ds))], tokenizer)
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        out = model(**batch)
    loss = out.loss.item()
    print(f"前向 OK，batch loss = {loss:.4f}")
    print("verify_train_setup 通过。可开始正式 train_pretrain.py / train_sft.py")


if __name__ == "__main__":
    main()
