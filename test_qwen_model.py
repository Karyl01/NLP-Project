"""
第 0 步：验证扩增词表后的 MedVocab 模型能否正常加载与生成。
用法（在项目根目录）:
  python test_qwen_model.py
  python test_qwen_model.py --model_path ./qwen2.5-1.5B-MedVocab
"""
import argparse
import os
import sys

import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL_PATH = "./qwen2.5-1.5B-MedVocab"

MEDICAL_QUESTION = (
    "你好！请问「半夏白术天麻汤」通常是由哪几味中药组成的？它的主治功效是什么？"
)


def parse_args():
    parser = argparse.ArgumentParser(description="MedVocab 模型推理冒烟测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="本地模型目录（默认：扩增词表后的 MedVocab）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--skip_generate", action="store_true", help="只测加载与分词，不生成")
    return parser.parse_args()


def check_tokenizer(tokenizer, label: str):
    sample = "半夏白术天麻汤、曲匹地尔片、沙利度胺"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    print(f"\n[{label}] 词表大小 vocab_size = {len(tokenizer)}")
    print(f"[{label}] 样例文本: {sample}")
    print(f"[{label}] token 数 = {len(ids)}  (越少说明专业词切分越短)")


def main():
    args = parse_args()
    model_path = os.path.abspath(args.model_path)

    print(f"模型路径: {model_path}")
    if not os.path.isdir(model_path):
        print(f"错误: 目录不存在 -> {model_path}")
        sys.exit(1)

    required = ["config.json", "tokenizer.json", "model.safetensors"]
    missing = [f for f in required if not os.path.isfile(os.path.join(model_path, f))]
    if missing:
        print(f"错误: 缺少文件 {missing}")
        sys.exit(1)

    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("模型与 Tokenizer 加载成功。")
    except Exception as e:
        print(f"加载失败:\n{e}")
        sys.exit(1)

    check_tokenizer(tokenizer, "MedVocab")

    if args.skip_generate:
        print("\n[--skip_generate] 已跳过生成，加载与分词检查通过。")
        return

    messages = [
        {"role": "system", "content": "你是一位专业的医疗 AI 助手。"},
        {"role": "user", "content": MEDICAL_QUESTION},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
    )

    print("\n[开始流式生成] ->\n")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        print(new_text, end="", flush=True)
    thread.join()

    print("\n\n第 0 步冒烟测试完成。")


if __name__ == "__main__":
    main()
