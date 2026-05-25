#!/usr/bin/env python3
"""
复现 stage2 在 step=500 触发的 eval 崩溃，用于定位是「验证批大小 / 标签 / 某条样本」问题。

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/debug_sft_eval.py \\
    --base_model_path ./qwen2.5-1.5B-MedVocab \\
    --adapter_path outputs/model_c/stage1_pretrain \\
    --valid_file data/sampled_sft_valid.jsonl
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import (
    data_collator,
    jsonl_to_dataset,
    load_model_for_continue,
    tokenize_sft_row,
    validate_sft_dataset,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--valid_file", default="data/sampled_sft_valid.jsonl")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model_for_continue(args.base_model_path, args.adapter_path)
    model.eval()

    raw = jsonl_to_dataset(args.valid_file)
    fn = partial(tokenize_sft_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    ds = raw.map(fn, remove_columns=raw.column_names)
    validate_sft_dataset(ds, tokenizer, "valid")

    collator = partial(data_collator, tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, collate_fn=collator)

    print(f"valid={len(ds)}, eval_batch_size={args.eval_batch_size}")
    for step, batch in enumerate(loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        loss = out.loss.item()
        print(f"  batch {step}: loss={loss:.4f}, shape={tuple(batch['input_ids'].shape)}")
    print("全部验证 batch 前向通过。")


if __name__ == "__main__":
    main()
