#!/usr/bin/env python3
"""
第 1 步自测：检查 data/ 下抽样文件是否齐全、格式是否正确。
用法: python scripts/verify_sampled_data.py
"""
import json
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_jsonl(path: Path, required_keys: set[str], max_preview: int = 1) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = required_keys - row.keys()
            if missing:
                raise ValueError(f"{path}: 缺少字段 {missing}")
            n += 1
            if n <= max_preview:
                pass
    return n


def main() -> None:
    root = project_root()
    data = root / "data"
    expected = {
        "sampled_pretrain.jsonl": {"text"},
        "sampled_sft_train.jsonl": {"instruction", "input", "output"},
        "sampled_sft_valid.jsonl": {"instruction", "input", "output"},
        "sampled_sft_test.jsonl": {"instruction", "input", "output"},
        "sampled_reward_train.jsonl": {
            "question",
            "response_chosen",
            "response_rejected",
        },
        "sampled_reward_valid.jsonl": {
            "question",
            "response_chosen",
            "response_rejected",
        },
        "sampled_reward_test.jsonl": {
            "question",
            "response_chosen",
            "response_rejected",
        },
    }

    manifest_path = data / "sample_manifest.json"
    if not manifest_path.is_file():
        print(f"错误: 请先运行 python scripts/sample_data.py，未找到 {manifest_path}")
        sys.exit(1)

    print(f"数据目录: {data}\n")
    all_ok = True
    for fname, keys in expected.items():
        path = data / fname
        try:
            n = check_jsonl(path, keys)
            print(f"  OK  {fname:30s}  {n:>8,} 条")
        except Exception as e:
            all_ok = False
            print(f"  FAIL {fname:30s}  {e}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"\nmanifest seed = {manifest.get('seed')}")
    print(f"manifest files = {list(manifest.get('files', {}).keys())}")

    if not all_ok:
        sys.exit(1)
    print("\n第 1 步数据校验通过。")


if __name__ == "__main__":
    main()
