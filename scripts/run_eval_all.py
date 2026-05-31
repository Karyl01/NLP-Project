#!/usr/bin/env python3
"""
按 configs/eval_models.json 批量推理。
先从 test 集按 sample_seed 随机抽 N 题（写入共享子集文件），6 个模型答同一批题。

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_eval_all.py
  python scripts/run_eval_all.py --max_samples 50 --sample_seed 123
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_inference import sample_test_rows  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "eval_models.json"))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--sample_seed", type=int, default=None)
    p.add_argument("--sequential", action="store_true", help="不随机，取 test 最前 N 条")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force_resample", action="store_true", help="强制重新生成随机子集")
    return p.parse_args()


def ensure_test_subset(test_file: Path, subset_path: Path, n: int, seed: int, sequential: bool, force: bool):
    if subset_path.is_file() and not force:
        print(f"复用已有子集: {subset_path}")
        return
    rows, pick = sample_test_rows(str(test_file), n, sample_seed=seed, sequential=sequential)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"source_file": str(test_file), "sample_seed": seed, "sequential": sequential, "indices": pick, "count": len(rows)}
    subset_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已生成随机子集 {len(rows)} 条 -> {subset_path}")
    print(f"  行号(0-based): {pick[:15]}{'...' if len(pick) > 15 else ''}")


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    test_file = ROOT / cfg["test_file"].lstrip("./")
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 100)
    seed = args.sample_seed if args.sample_seed is not None else cfg.get("sample_seed", 42)
    sequential = args.sequential
    results_dir = ROOT / cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if sequential:
        subset_path = results_dir / f"test_subset_sequential_n{max_samples}.jsonl"
    else:
        subset_path = results_dir / f"test_subset_seed{seed}_n{max_samples}.jsonl"

    ensure_test_subset(test_file, subset_path, max_samples, seed, sequential, args.force_resample)

    models = cfg["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["id"] in wanted]

    manifest = {
        "test_subset": str(subset_path.relative_to(ROOT)),
        "sample_seed": seed,
        "sequential": sequential,
        "max_samples": max_samples,
        "predictions": {},
    }

    print(f"\n共 {len(models)} 个模型，统一 test 子集: {subset_path.name}\n")

    for m in models:
        mid = m["id"]
        out_file = results_dir / f"{mid}_test_pred.jsonl"
        manifest["predictions"][mid] = str(out_file.relative_to(ROOT))
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_inference.py"),
            "--base_model_path",
            str(ROOT / m["base_model_path"].lstrip("./")),
            "--test_file",
            str(subset_path),
            "--max_samples",
            str(max_samples),
            "--sequential",
            "--output_file",
            str(out_file),
        ]
        adapter = m.get("adapter_path")
        if adapter:
            cmd.extend(["--adapter_path", str(ROOT / str(adapter).lstrip("./"))])
        print(f">>> [{mid}] {m['label']}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        ret = subprocess.run(cmd, cwd=str(ROOT))
        if ret.returncode != 0:
            sys.exit(ret.returncode)
        print(f"    -> {out_file}\n")

    (results_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"下一步: python scripts/build_match_results.py --config {args.config} "
        f"&& python llm_judge.py --mode all --config {args.config} --match_file <合并后的json>"
    )


if __name__ == "__main__":
    main()
