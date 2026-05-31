#!/usr/bin/env python3
"""
第 1 步：从 medical 全量数据中抽取可复现的训练子集。

产出目录（默认 data/）:
  sampled_pretrain.jsonl   - 领域激活 / DAPT（模型 C 阶段1、模型 B）
  sampled_sft_train.jsonl  - 指令 SFT 训练（A/B/C 共用，须相同种子）
  sampled_sft_valid.jsonl  - SFT 验证（完整 valid_zh_0，500 条）
  sampled_sft_test.jsonl   - 评测集（完整 test_zh_0，500 条，训练时勿混入）
  sampled_reward_*.jsonl - 可选（本项目不做 DPO，可忽略）

用法（在项目根目录）:
  python scripts/sample_data.py
  python scripts/sample_data.py --pretrain_encyclopedia_n 20000 --sft_train_n 30000

  # 组长实验：train_zh_0 顺序前 5 万条（非随机）
  python scripts/sample_data.py --only_sft_head 50000 --skip_pretrain
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {e}") from e


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def head_sample_jsonl(path: Path, k: int) -> list[dict[str, Any]]:
    """取 JSONL 文件最前面的 k 条（顺序、可复现）。"""
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if len(rows) >= k:
            break
    if len(rows) < k:
        print(
            f"  警告: {path.name} 仅 {len(rows)} 行，少于请求 {k} 条，将全部保留。",
            file=sys.stderr,
        )
    return rows


def reservoir_sample_jsonl(
    path: Path, k: int, rng: random.Random
) -> list[dict[str, Any]]:
    """对 JSONL 做 reservoir sampling，单遍扫描，适合超大文件。"""
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for row in iter_jsonl(path):
        seen += 1
        if len(reservoir) < k:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen - 1)
            if j < k:
                reservoir[j] = row
    if seen < k:
        print(
            f"  警告: {path.name} 仅 {seen} 行，少于请求抽样数 {k}，将全部保留。",
            file=sys.stderr,
        )
    return reservoir


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_jsonl(src: Path, dst: Path) -> int:
    rows = list(iter_jsonl(src))
    write_jsonl(dst, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    root = project_root()
    medical = root / "medical"
    p = argparse.ArgumentParser(description="抽取医疗训练数据子集（固定随机种子）")
    p.add_argument("--seed", type=int, default=42, help="随机种子，A/B/C 实验须一致")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=root / "data",
        help="输出目录",
    )
    p.add_argument(
        "--pretrain_encyclopedia_n",
        type=int,
        default=30_000,
        help="从 train_encyclopedia.json 随机抽取条数",
    )
    p.add_argument(
        "--sft_train_n",
        type=int,
        default=50_000,
        help="从 finetune/train_zh_0.json 随机抽取条数",
    )
    p.add_argument(
        "--skip_pretrain",
        action="store_true",
        help="若已生成 pretrain 子集可跳过",
    )
    p.add_argument(
        "--skip_sft",
        action="store_true",
        help="若已生成 SFT 子集可跳过",
    )
    p.add_argument(
        "--only_sft_head",
        type=int,
        default=None,
        metavar="N",
        help="仅导出 train_zh_0 前 N 条到 sampled_sft_train_head{N}.jsonl（顺序取样）",
    )
    p.add_argument(
        "--skip_reward",
        action="store_true",
        help="跳过 reward 复制",
    )
    p.add_argument(
        "--medical_dir",
        type=Path,
        default=medical,
        help="medical 数据集根目录",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    medical = args.medical_dir.resolve()
    out_dir = args.out_dir.resolve()

    paths = {
        "pretrain_book": medical / "pretrain" / "medical_book_zh.json",
        "pretrain_ency": medical / "pretrain" / "train_encyclopedia.json",
        "sft_train": medical / "finetune" / "train_zh_0.json",
        "sft_valid": medical / "finetune" / "valid_zh_0.json",
        "sft_test": medical / "finetune" / "test_zh_0.json",
        "reward_train": medical / "reward" / "train.json",
        "reward_valid": medical / "reward" / "valid.json",
        "reward_test": medical / "reward" / "test.json",
    }
    for name, p in paths.items():
        if not p.is_file():
            print(f"错误: 找不到 {name} -> {p}")
            sys.exit(1)

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "out_dir": str(out_dir),
        "files": {},
    }

    print(f"输出目录: {out_dir}")
    print(f"随机种子: {args.seed}\n")

    if args.only_sft_head is not None:
        n = args.only_sft_head
        print(f"[SFT head] 从 train_zh_0 顺序取前 {n} 条 ...")
        rows = head_sample_jsonl(paths["sft_train"], n)
        out_head = out_dir / f"sampled_sft_train_head{n}.jsonl"
        write_jsonl(out_head, rows)
        manifest["files"][out_head.name] = {
            "count": len(rows),
            "source": str(paths["sft_train"]),
            "head_sequential": True,
            "head_n": n,
        }
        manifest_path = out_dir / "sample_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_head}  共 {len(rows)} 条")
        print(f"清单: {manifest_path}")
        print("\n完成。请用此文件跑 train_sft.py（见脚本顶部组长实验说明）。")
        return

    # ----- Pretrain 子集 -----
    if not args.skip_pretrain:
        print("[1/3] 构建 pretrain 子集 ...")
        book_rows = list(iter_jsonl(paths["pretrain_book"]))
        print(f"  medical_book_zh.json: 全量 {len(book_rows)} 条")

        print(
            f"  正在从 train_encyclopedia.json 抽样 {args.pretrain_encyclopedia_n} 条 "
            f"（大文件，请耐心等待）..."
        )
        ency_rows = reservoir_sample_jsonl(
            paths["pretrain_ency"], args.pretrain_encyclopedia_n, rng
        )
        print(f"  train_encyclopedia 抽样完成: {len(ency_rows)} 条")

        pretrain_rows = book_rows + ency_rows
        rng.shuffle(pretrain_rows)
        out_pretrain = out_dir / "sampled_pretrain.jsonl"
        write_jsonl(out_pretrain, pretrain_rows)
        manifest["files"]["sampled_pretrain.jsonl"] = {
            "count": len(pretrain_rows),
            "book_full": len(book_rows),
            "encyclopedia_sampled": len(ency_rows),
        }
        print(f"  -> {out_pretrain}  共 {len(pretrain_rows)} 条\n")
    else:
        print("[1/3] 跳过 pretrain\n")

    # ----- SFT 子集 -----
    if not args.skip_sft:
        print("[2/3] 构建 SFT 子集 ...")
        print(
            f"  正在从 train_zh_0.json 抽样 {args.sft_train_n} 条 "
            f"（约 195 万行，耗时较长）..."
        )
        sft_train = reservoir_sample_jsonl(paths["sft_train"], args.sft_train_n, rng)
        out_sft_train = out_dir / "sampled_sft_train.jsonl"
        write_jsonl(out_sft_train, sft_train)
        manifest["files"]["sampled_sft_train.jsonl"] = {
            "count": len(sft_train),
            "source": str(paths["sft_train"]),
        }
        print(f"  -> {out_sft_train}  {len(sft_train)} 条")

        for split, src_key, out_name in [
            ("valid", "sft_valid", "sampled_sft_valid.jsonl"),
            ("test", "sft_test", "sampled_sft_test.jsonl"),
        ]:
            n = copy_jsonl(paths[src_key], out_dir / out_name)
            manifest["files"][out_name] = {"count": n, "copied_full": True}
            print(f"  -> {out_dir / out_name}  {n} 条（完整复制，{split}）")
        print()
    else:
        print("[2/3] 跳过 SFT\n")

    # ----- Reward（DPO）全量复制 -----
    if args.skip_reward:
        print("[3/3] 跳过 reward\n")
    else:
        print("[3/3] 复制 reward 数据（体量小，全量）...")
        for split, src_key, out_name in [
            ("train", "reward_train", "sampled_reward_train.jsonl"),
            ("valid", "reward_valid", "sampled_reward_valid.jsonl"),
            ("test", "reward_test", "sampled_reward_test.jsonl"),
        ]:
            n = copy_jsonl(paths[src_key], out_dir / out_name)
            manifest["files"][out_name] = {"count": n, "copied_full": True}
            print(f"  -> {out_dir / out_name}  {n} 条")
        print()

    manifest_path = out_dir / "sample_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"清单已写入: {manifest_path}")
    print("\n第 1 步数据准备完成。请将 sample_manifest.json 中的 seed 记入实验记录。")


if __name__ == "__main__":
    main()
