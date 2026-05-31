"""训练脚本共用：模型加载、LoRA、数据集与 SFT 标签构造。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class TrainPaths:
    model_path: str
    train_file: str
    output_dir: str
    valid_file: str | None = None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsonl_to_dataset(path: str | Path) -> Dataset:
    return Dataset.from_list(load_jsonl(path))


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_path: str,
    use_4bit: bool = False,
    use_gradient_checkpointing: bool = True,
):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant_config,
    )
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def build_lora_model(
    model,
    save_embeddings: bool = False,
    lora_r: int = 64,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
):
    """
    save_embeddings=True：用于扩增词表后的 MedVocab，训练 embed_tokens + lm_head。
    save_embeddings=False：基线 A/B 的 DAPT/SFT，仅 LoRA。
    """
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    modules_to_save = None
    if save_embeddings:
        modules_to_save = ["embed_tokens", "lm_head"]

    if lora_alpha is None:
        lora_alpha = lora_r * 2

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    return get_peft_model(model, config)


def load_model_for_continue(
    base_model_path: str,
    adapter_path: str,
    use_4bit: bool = False,
):
    """从 stage1 adapter 继续 stage2 SFT。"""
    tokenizer = load_tokenizer(adapter_path if (Path(adapter_path) / "tokenizer_config.json").exists() else base_model_path)
    model = load_base_model(base_model_path, use_4bit=use_4bit)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model, tokenizer


def format_pretrain_text(example: dict[str, Any]) -> str:
    return example["text"].strip()


def format_sft_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    instruction = (example.get("instruction") or "").strip()
    user_input = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    user_content = instruction
    if user_input:
        user_content = f"{instruction}\n{user_input}" if instruction else user_input
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def tokenize_pretrain_row(example: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    text = format_pretrain_text(example)
    out = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    out["labels"] = out["input_ids"].copy()
    return out


def get_tokenizer_vocab_size(tokenizer) -> int:
    """
    扩表后 tokenizer.vocab_size 可能仍是旧值（如 151643），
    必须以 len(tokenizer) 为准，否则会误报非法 token。
    """
    return len(tokenizer)


def _assistant_prefix_len(tokenizer, full_text: str, max_seq_length: int) -> int:
    """在整段 chat 文本里定位 assistant 回答起始 token 位置（避免两次 truncate 不对齐）。"""
    markers = ("<|im_start|>assistant\n", "<|im_start|>assistant")
    prefix_text = None
    for m in markers:
        if m in full_text:
            prefix_text = full_text[: full_text.index(m) + len(m)]
            break
    if prefix_text is None:
        return 0
    return len(
        tokenizer(
            prefix_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]
    )


def tokenize_sft_row(example: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    messages = format_sft_messages(example)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    input_ids = enc["input_ids"]
    vocab_size = get_tokenizer_vocab_size(tokenizer)

    labels = input_ids.copy()
    prompt_len = _assistant_prefix_len(tokenizer, full_text, max_seq_length)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    for tid in input_ids:
        if tid < 0 or tid >= vocab_size:
            raise ValueError(
                f"非法 input_id={tid}, len(tokenizer)={vocab_size}, "
                f"tokenizer.vocab_size={getattr(tokenizer, 'vocab_size', None)}"
            )

    return {"input_ids": input_ids, "labels": labels}


def validate_sft_dataset(dataset, tokenizer, name: str = "dataset") -> None:
    """扫描数据，提前发现超长或非法 token。"""
    vocab_size = get_tokenizer_vocab_size(tokenizer)
    print(
        f"[{name}] len(tokenizer)={vocab_size}, "
        f"tokenizer.vocab_size={getattr(tokenizer, 'vocab_size', None)}"
    )
    max_len = 0
    bad = 0
    for i in range(len(dataset)):
        row = dataset[i]
        ids = row["input_ids"]
        max_len = max(max_len, len(ids))
        for tid in ids:
            if tid < 0 or tid >= vocab_size:
                bad += 1
                print(f"[{name}] 样本 {i} 非法 token id={tid}")
                break
        for lab in row["labels"]:
            if lab != -100 and (lab < 0 or lab >= vocab_size):
                bad += 1
                print(f"[{name}] 样本 {i} 非法 label id={lab}")
                break
    print(f"[{name}] 条数={len(dataset)}, max_len={max_len}, 异常样本={bad}")


def print_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0
    print(f"可训练参数: {trainable:,} / {total:,} ({pct:.4f}%)")


def data_collator(features: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    """动态 padding；labels 中 -100 保持。"""
    batch = tokenizer.pad(
        {"input_ids": [f["input_ids"] for f in features]},
        padding=True,
        return_tensors="pt",
    )
    labels = [f["labels"] for f in features]
    max_len = batch["input_ids"].shape[1]
    padded_labels = []
    for lab in labels:
        padded = lab + [-100] * (max_len - len(lab))
        padded_labels.append(padded[:max_len])
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    # 保持 long 类型，部分 CUDA kernel 对 bool mask 更敏感
    batch["attention_mask"] = batch["attention_mask"].to(dtype=torch.long)
    return batch
