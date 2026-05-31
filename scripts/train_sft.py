#!/usr/bin/env python3
"""
第 3 步（阶段 2）：指令 SFT。A/B/C 三个对比模型共用同一套 SFT 数据（data/sampled_sft_train.jsonl）。

用法示例:
  # 模型 C：接 stage1 adapter（重训 SFT 加 --fresh_lora）
  python scripts/train_sft.py \\
    --base_model_path ./qwen2.5-1.5B-MedVocab \\
    --adapter_path outputs/model_c/stage1_pretrain \\
    --train_file data/sampled_sft_train.jsonl \\
    --output_dir outputs/model_c/stage2_sft \\
    --save_embeddings \\
    --fresh_lora

  # 模型 A：原始 Qwen，仅 SFT
  python scripts/train_sft.py \\
    --model_path ./models/qwen2.5-1.5B-Instruct \\
    --train_file data/sampled_sft_train.jsonl \\
    --output_dir outputs/model_a/sft

  # 模型 B：接 DAPT adapter（重训 SFT 时用 --fresh_lora 合并 stage1 后新建 r=64 LoRA）
  python scripts/train_sft.py \\
    --base_model_path ./models/qwen2.5-1.5B-Instruct \\
    --adapter_path outputs/model_b/stage1_dapt \\
    --train_file data/sampled_sft_train.jsonl \\
    --output_dir outputs/model_b/stage2_sft \\
    --fresh_lora

  # 模型 C'：MedVocab 仅 SFT
  python scripts/train_sft.py \\
    --model_path ./qwen2.5-1.5B-MedVocab \\
    --train_file data/sampled_sft_train.jsonl \\
    --output_dir outputs/model_c_sft_only/sft \\
    --save_embeddings

  # ----- 组长实验：train_zh_0 前 5 万条（先 python scripts/sample_data.py --only_sft_head 50000 --skip_pretrain）-----
  # 1) 原始 Instruct 底模 + LoRA SFT
  CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py \\
    --model_path ./models/qwen2.5-1.5B-Instruct \\
    --train_file data/sampled_sft_train_head50000.jsonl \\
    --output_dir outputs/instruct_head50k/sft

  # 2) 组长全量模型 qwen2.5-1.5B-finetune-train-Medvocab + LoRA SFT
  CUDA_VISIBLE_DEVICES=1 python scripts/train_sft.py \\
    --model_path ./qwen2.5-1.5B-finetune-train-Medvocab \\
    --train_file data/sampled_sft_train_head50000.jsonl \\
    --output_dir outputs/leader_medvocab_head50k/sft \\
    --save_embeddings
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
    get_tokenizer_vocab_size,
    jsonl_to_dataset,
    load_base_model,
    load_model_for_continue,
    load_tokenizer,
    print_trainable_parameters,
    tokenize_sft_row,
    validate_sft_dataset,
)


def check_tokenizer_model_alignment(model, tokenizer) -> None:
    tok_n = get_tokenizer_vocab_size(tokenizer)
    model_n = model.config.vocab_size
    print(f"词表对齐检查: len(tokenizer)={tok_n}, model.config.vocab_size={model_n}")
    if tok_n != model_n:
        print(
            "警告: tokenizer 与模型 vocab 不一致，请确认 adapter 与底模来自同一 MedVocab。"
        )


def parse_args():
    p = argparse.ArgumentParser(description="指令 SFT")
    p.add_argument("--model_path", type=str, default=None, help="从头训练时指定（模型 A）")
    p.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="接 stage1 时底模路径（通常与 stage1 相同）",
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="stage1 输出目录；若设置则从该 adapter 继续",
    )
    p.add_argument("--train_file", type=str, default="data/sampled_sft_train.jsonl")
    p.add_argument(
        "--valid_file",
        type=str,
        default=None,
        help="验证集路径；默认 None=不加载，避免训练中 eval",
    )
    p.add_argument("--output_dir", type=str, default="outputs/model_c/stage2_sft")
    p.add_argument("--save_embeddings", action="store_true", help="MedVocab 模型 C 建议开启")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="默认 2×lora_r（r=64 时为 128）",
    )
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument(
        "--enable_eval",
        action="store_true",
        help="训练时跑验证（默认关闭；原脚本在 save_steps=500 的 eval 处稳定崩溃）",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="断点续训：checkpoint 路径，或 auto",
    )
    p.add_argument(
        "--no_valid",
        action="store_true",
        help="不使用验证集（与默认不 eval 配合，彻底跳过 valid 加载）",
    )
    p.add_argument(
        "--fresh_lora",
        action="store_true",
        help="合并 stage1 adapter 到基座后新建 LoRA（重训 SFT 且 r 与 stage1 不同时必开）",
    )
    return p.parse_args()


def resolve_resume_checkpoint(output_dir: str, resume_arg: str | None):
    if not resume_arg:
        return None
    out = Path(output_dir)
    if resume_arg == "auto":
        checkpoints = sorted(
            out.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        return str(checkpoints[-1]) if checkpoints else None
    return resume_arg


def main():
    args = parse_args()
    print("=== 阶段 2: SFT ===")

    if args.adapter_path:
        base = args.base_model_path or args.model_path
        if not base:
            raise ValueError("使用 --adapter_path 时必须提供 --base_model_path 或 --model_path")
        print(f"从 adapter 继续: {args.adapter_path}")
        print(f"底模: {base}")
        model, tokenizer = load_model_for_continue(base, args.adapter_path, use_4bit=args.use_4bit)
        check_tokenizer_model_alignment(model, tokenizer)
        if not args.fresh_lora:
            print(
                "提示: 未加 --fresh_lora 时将在 stage1 原 LoRA 上继续训（rank 与 stage1 一致）。"
                "重训 SFT 且使用新 r 时请加上 --fresh_lora。"
            )
        if args.fresh_lora:
            print(f"合并 stage1 adapter 后新建 LoRA: r={args.lora_r}, alpha={args.lora_alpha or args.lora_r * 2}")
            model = model.merge_and_unload()
            model = build_lora_model(
                model,
                save_embeddings=args.save_embeddings,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
    else:
        if not args.model_path:
            raise ValueError("请指定 --model_path 或 --adapter_path")
        print(f"model_path: {args.model_path}")
        tokenizer = load_tokenizer(args.model_path)
        model = load_base_model(args.model_path, use_4bit=args.use_4bit)
        model = build_lora_model(
            model,
            save_embeddings=args.save_embeddings,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        check_tokenizer_model_alignment(model, tokenizer)

    print_trainable_parameters(model)
    print(f"train_file     : {args.train_file}")
    print(f"output_dir     : {args.output_dir}")
    print(f"learning_rate  : {args.learning_rate}")
    print(f"lora_r/alpha   : {args.lora_r} / {args.lora_alpha or args.lora_r * 2}")

    dataset = jsonl_to_dataset(args.train_file)
    print(f"训练样本数: {len(dataset)}")

    tokenize_fn = partial(tokenize_sft_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    dataset = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="tokenize sft",
    )
    validate_sft_dataset(dataset.select(range(min(500, len(dataset)))), tokenizer, "train(sample)")

    eval_dataset = None
    if args.valid_file and not args.no_valid:
        raw_eval = jsonl_to_dataset(args.valid_file)
        eval_dataset = raw_eval.map(
            tokenize_fn,
            remove_columns=raw_eval.column_names,
            desc="tokenize valid",
        )
        validate_sft_dataset(eval_dataset, tokenizer, "valid")

    use_eval = bool(args.enable_eval and eval_dataset is not None)
    if eval_dataset is not None and not args.enable_eval:
        print(
            "已加载 valid 但默认不在训练中 eval（避免 step=500 处 CUDA 崩溃）。"
            "训练后用 test 集评测；若坚持训练中验证请加 --enable_eval"
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.save_steps if use_eval else None,
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    resume_ckpt = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_ckpt:
        print(f"从 checkpoint 续训: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n阶段 2 SFT 完成: {args.output_dir}")


if __name__ == "__main__":
    main()
