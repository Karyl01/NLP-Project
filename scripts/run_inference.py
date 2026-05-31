#!/usr/bin/env python3
"""
加载「底模 + LoRA adapter」做医疗问答推理 / 批量评测。

用法（项目根目录）:

  # 1) 单条交互式冒烟（模型 C）
  CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py \\
    --base_model_path ./qwen2.5-1.5B-MedVocab \\
    --adapter_path outputs/model_c/stage2_sft \\
    --question "曲匹地尔片的用法用量是多少？"

  # 2) 在测试集上跑 N 条，结果写入 jsonl
  CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py \\
    --base_model_path ./qwen2.5-1.5B-MedVocab \\
    --adapter_path outputs/model_c/stage2_sft \\
    --test_file data/sampled_sft_test.jsonl \\
    --max_samples 20 \\
    --output_file results/model_c_predictions.jsonl

  # 3) 只测底模（未微调），不传 adapter_path
  CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py \\
    --base_model_path ./qwen2.5-1.5B-MedVocab \\
    --question "半夏白术天麻汤由哪些药组成？"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import format_sft_messages


def parse_args():
    p = argparse.ArgumentParser(description="医疗模型推理与批量测试")
    p.add_argument("--base_model_path", type=str, default="./qwen2.5-1.5B-MedVocab")
    p.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA 输出目录，如 outputs/model_c/stage2_sft；不传则只用底模",
    )
    p.add_argument("--question", type=str, default=None, help="单条问题")
    p.add_argument("--test_file", type=str, default=None, help="SFT 测试 jsonl")
    p.add_argument("--max_samples", type=int, default=10, help="批量测试最多条数")
    p.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="从 test 全集随机抽样；与 run_eval_all 共用 seed 可保证各模型同一批题",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="不随机，始终取文件最前面的 max_samples 条",
    )
    p.add_argument("--output_file", type=str, default=None, help="批量预测保存路径")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument(
        "--system_prompt",
        type=str,
        default="你是一位专业的医疗 AI 助手。",
    )
    return p.parse_args()


def load_model_and_tokenizer(base_path: str, adapter_path: str | None):
    tok_dir = adapter_path if adapter_path and (Path(adapter_path) / "tokenizer.json").exists() else base_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if adapter_path:
        print(f"加载 adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def build_user_text(example: dict) -> str:
    instruction = (example.get("instruction") or "").strip()
    user_input = (example.get("input") or "").strip()
    if user_input:
        return f"{instruction}\n{user_input}" if instruction else user_input
    return instruction


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    user_text: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - t0

    new_tokens = out[0, prompt_tokens:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, int(new_tokens.shape[0]), elapsed


def load_all_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_test_rows(
    path: str,
    max_samples: int,
    sample_seed: int | None = 42,
    sequential: bool = False,
) -> tuple[list[dict], list[int]]:
    """返回 (样本列表, 在全集中的行号)。sequential=True 时取前 N 条。"""
    all_rows = load_all_jsonl(path)
    n = min(max_samples, len(all_rows))
    if sequential or sample_seed is None:
        pick = list(range(n))
    else:
        indices = list(range(len(all_rows)))
        random.Random(sample_seed).shuffle(indices)
        pick = sorted(indices[:n])
    return [all_rows[i] for i in pick], pick


def load_jsonl(path: str, limit: int, sample_seed: int | None = None, sequential: bool = False) -> list[dict]:
    rows, _ = sample_test_rows(path, limit, sample_seed=sample_seed, sequential=sequential)
    return rows


def main():
    args = parse_args()
    if not args.question and not args.test_file:
        print("请指定 --question 或 --test_file")
        sys.exit(1)

    print(f"底模: {args.base_model_path}")
    print(f"adapter: {args.adapter_path or '(无，仅底模)'}")
    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path)

    if args.question:
        ans, n_tok, sec = generate_answer(
            model,
            tokenizer,
            args.question,
            args.system_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        print(f"\n【问题】\n{args.question}\n")
        print(f"【回答】\n{ans}\n")
        print(f"生成 token 数: {n_tok}, 耗时: {sec:.2f}s, tokens/s: {n_tok/sec:.1f}")
        return

    rows, pick_idx = sample_test_rows(
        args.test_file,
        args.max_samples,
        sample_seed=args.sample_seed,
        sequential=args.sequential,
    )
    mode = "顺序前N条" if args.sequential else f"随机抽样 seed={args.sample_seed}"
    print(f"测试文件: {args.test_file}, 样本数: {len(rows)} ({mode})")
    print(f"全集行号(0-based): {pick_idx[:10]}{'...' if len(pick_idx) > 10 else ''}")

    results = []
    total_new_tokens = 0
    total_time = 0.0

    for i, ex in enumerate(rows):
        user_text = build_user_text(ex)
        ref = (ex.get("output") or "").strip()
        pred, n_tok, sec = generate_answer(
            model,
            tokenizer,
            user_text,
            args.system_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        total_new_tokens += n_tok
        total_time += sec
        row = {
            "index": i,
            "source_line": pick_idx[i],
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "reference": ref,
            "prediction": pred,
            "new_tokens": n_tok,
            "latency_sec": round(sec, 3),
        }
        results.append(row)
        print(f"\n===== [{i+1}/{len(rows)}] =====")
        print(f"问: {user_text[:120]}{'...' if len(user_text) > 120 else ''}")
        print(f"参考: {ref[:120]}{'...' if len(ref) > 120 else ''}")
        print(f"预测: {pred[:120]}{'...' if len(pred) > 120 else ''}")

    avg_tps = total_new_tokens / total_time if total_time > 0 else 0
    print(f"\n汇总: {len(rows)} 条, 平均 tokens/s = {avg_tps:.1f}")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"预测已保存: {out_path}")


if __name__ == "__main__":
    main()
