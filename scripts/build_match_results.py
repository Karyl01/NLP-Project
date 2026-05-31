#!/usr/bin/env python3
"""
合并 configs 中全部模型的预测 jsonl 为统一评测表（供 llm_judge --mode all）。

用法:
  python scripts/build_match_results.py
  python scripts/build_match_results.py --config configs/eval_models.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_user_text(row: dict) -> str:
    instruction = (row.get("instruction") or "").strip()
    user_input = (row.get("input") or "").strip()
    if user_input:
        return f"{instruction}\n{user_input}" if instruction else user_input
    return instruction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "eval_models.json"))
    p.add_argument("--output", default=str(ROOT / "match_eval_results_all.json"))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results_dir = ROOT / cfg.get("results_dir", "results")

    # id -> (output_field, pred rows)
    loaded: dict[str, tuple[str, list[dict]]] = {}
    for m in cfg["models"]:
        mid = m["id"]
        pred_path = results_dir / f"{mid}_test_pred.jsonl"
        if not pred_path.is_file():
            raise FileNotFoundError(f"缺少预测文件: {pred_path}，请先 run_eval_all.py")
        loaded[mid] = (m["output_field"], load_jsonl(pred_path))
        print(f"  {mid}: {len(loaded[mid][1])} 条 <- {pred_path.name}")

    n = len(next(iter(loaded.values()))[1])
    for mid, (_, rows) in loaded.items():
        if len(rows) != n:
            raise ValueError(f"{mid} 条数 {len(rows)} != {n}")

    match = []
    first_rows = loaded[cfg["models"][0]["id"]][1]
    for i in range(n):
        gold = (first_rows[i].get("reference") or first_rows[i].get("output") or "").strip()
        prompt = build_user_text(first_rows[i])
        row = {
            "id": i + 1,
            "patient_prompt": prompt,
            "dataset_gold_output": gold,
        }
        for m in cfg["models"]:
            mid = m["id"]
            field = m["output_field"]
            pred_row = loaded[mid][1][i]
            if build_user_text(pred_row) != prompt:
                print(f"警告: 样本 {i+1} 在 {mid} 上问题文本不一致")
            row[field] = (pred_row.get("prediction") or "").strip()
            row[f"{mid}_new_tokens"] = pred_row.get("new_tokens")
            row[f"{mid}_latency_sec"] = pred_row.get("latency_sec")
        match.append(row)

    out = Path(args.output)
    payload = {
        "_meta": {
            "model_ids": [m["id"] for m in cfg["models"]],
            "num_cases": len(match),
        },
        "cases": match,
    }
    manifest_path = results_dir / "eval_manifest.json"
    if manifest_path.is_file():
        payload["_meta"]["eval_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta_files = sorted(results_dir.glob("test_subset_*.meta.json"))
    if meta_files:
        payload["_meta"]["test_subset"] = json.loads(meta_files[-1].read_text(encoding="utf-8"))

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}，共 {len(match)} 条 × {len(cfg['models'])} 模型")

    # token 效率汇总
    summary = {}
    for m in cfg["models"]:
        mid = m["id"]
        tokens = [loaded[mid][1][i].get("new_tokens", 0) for i in range(n)]
        lat = [loaded[mid][1][i].get("latency_sec", 0) for i in range(n)]
        total_tok = sum(tokens)
        total_lat = sum(lat)
        summary[mid] = {
            "label": m["label"],
            "avg_new_tokens": round(total_tok / n, 2),
            "avg_tokens_per_sec": round(total_tok / total_lat, 2) if total_lat else 0,
        }
    summary_path = results_dir / "eval_token_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"token 汇总: {summary_path}")
    print("下一步: python llm_judge.py --mode all")


if __name__ == "__main__":
    main()
