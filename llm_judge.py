#!/usr/bin/env python3
"""
LLM 盲审：configs/eval_models.json

  python llm_judge.py --mode all          # 6 模型同题对比（推荐）
  python llm_judge.py --mode pairs        # 配置中的两两对战
  python llm_judge.py --mode pair --left C_sft --right A_sft
  python llm_judge.py --mode pair --left leader_head50k --right instruct_head50k \\
    --config configs/eval_head50k.json --match_file match_eval_head50k.json --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

def _judge_client() -> OpenAI:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "请设置环境变量 ARK_API_KEY（火山方舟 API Key），例如: export ARK_API_KEY='your-key'"
        )
    return OpenAI(
        api_key=key,
        base_url=os.environ.get(
            "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3"
        ),
    )


client = None  # 延迟初始化，见 call_judge_llm
JUDGE_MODEL = "DeepSeek-V4-Flash"
# 火山部分模型不支持 response_format=json_object，默认关闭，从回复文本解析 JSON
USE_JSON_RESPONSE_FORMAT = False


def parse_json_from_text(text: str) -> dict:
    """从模型回复中解析 JSON（兼容 markdown 代码块）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("裁判返回为空")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法解析 JSON: {text[:200]}...")


def call_judge_llm(prompt: str) -> dict:
    global client
    if client is None:
        client = _judge_client()
    kwargs = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if USE_JSON_RESPONSE_FORMAT:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        err = str(e)
        if USE_JSON_RESPONSE_FORMAT and "json_object" in err:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    content = response.choices[0].message.content
    return parse_json_from_text(content)

JUDGE_PAIR_TEMPLATE = """
[系统角色]
你是一位资深的中文全科医学教授。请在完全不知道模型名称（全盲测试）的前提下，
对两个大模型针对患者提问的回答进行严格审评。

[患者提问]:
{patient_prompt}

[临床医生给出的金标准答案 (Gold Standard)]:
{gold_output}

---
[模型回答 1]:
{model_1_output}

---
[模型回答 2]:
{model_2_output}

---
[审评要求]:
请参考金标准，从以下三个维度对 [模型回答 1] 和 [模型回答 2] 全盲打分（每维 1-10 分）：
1. 医学准确性（无幻觉、大方向正确）
2. 术语规范性（医学术语使用是否恰当）
3. 临床实用性（条理清晰、适合医患沟通）

请严格输出 JSON，不要 Markdown：
{{
    "reasoning": "100字以内对比分析",
    "model_1_scores": {{"accuracy": 0, "terminology": 0, "practicality": 0}},
    "model_2_scores": {{"accuracy": 0, "terminology": 0, "practicality": 0}},
    "winner": "Model 1" 或 "Model 2" 或 "Tie"
}}
"""


def build_all_judge_prompt(n_models: int, patient_prompt: str, gold: str, outputs: list[str]) -> str:
    blocks = []
    for i, text in enumerate(outputs, start=1):
        blocks.append(f"---\n[模型回答 {i}]:\n{text}\n")
    score_lines = ",\n".join(
        [
            f'    "Model {i}": {{"accuracy": 0, "terminology": 0, "practicality": 0}}'
            for i in range(1, n_models + 1)
        ]
    )
    rank_example = ", ".join([f'"Model {i}"' for i in range(1, n_models + 1)])
    return f"""
[系统角色]
你是一位资深的中文全科医学教授。请在完全不知道模型名称（全盲测试）的前提下，
同时对 {n_models} 个大模型的回答进行审评与排序。

[患者提问]:
{patient_prompt}

[临床医生给出的金标准答案 (Gold Standard)]:
{gold}

{"".join(blocks)}
---
[审评要求]:
请参考金标准，从医学准确性、术语规范性、临床实用性三个维度，
为每个 [模型回答 k] 打分（每维 1-10），并给出综合排名（最好 -> 最差）。

请严格输出 JSON，不要 Markdown：
{{
    "reasoning": "150字以内总评",
    "scores": {{
{score_lines}
    }},
    "ranking": [{rank_example}],
    "best": "Model 1"
}}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "eval_models.json"))
    p.add_argument("--match_file", default=str(ROOT / "match_eval_results_all.json"))
    p.add_argument("--output_dir", default=str(ROOT / "results" / "judge"))
    p.add_argument("--mode", choices=["all", "pair", "pairs"], default="all")
    p.add_argument("--left", default=None)
    p.add_argument("--right", default=None)
    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="pair 模式：同一批样例重复盲审次数（每次独立随机左右顺序）",
    )
    p.add_argument(
        "--json_response_format",
        action="store_true",
        help="请求 API 使用 response_format=json_object（部分模型不支持）",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_match_file(path: Path) -> tuple[dict | None, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cases" in raw:
        return raw.get("_meta"), raw["cases"]
    if isinstance(raw, list):
        return None, raw
    raise ValueError("match 文件格式应为 list 或 {{cases: [...]}}")


def score_total(scores: dict) -> float:
    if not scores:
        return 0.0
    if scores.get("total"):
        return float(scores["total"])
    return sum(float(scores.get(k, 0)) for k in ("accuracy", "terminology", "practicality"))


def id_to_field(cfg: dict) -> dict[str, str]:
    return {m["id"]: m["output_field"] for m in cfg["models"]}


def output_fields_in_case(case: dict) -> list[str]:
    return [k for k in case if k.startswith("model_") and k.endswith("_output")]


def resolve_field(field_map: dict[str, str], case: dict, model_id: str) -> str:
    if model_id in field_map:
        return field_map[model_id]
    for key in output_fields_in_case(case):
        if model_id in key:
            return key
    raise KeyError(f"找不到模型 [{model_id}]，可用列: {output_fields_in_case(case)}")


def parse_model_label(label: str) -> int | None:
    m = re.match(r"Model\s*(\d+)", str(label).strip(), re.I)
    return int(m.group(1)) if m else None


def judge_one_pair(case, ans_left, ans_right, name_left, name_right, swap: bool):
    if swap:
        m1_out, m2_out = ans_right, ans_left
        m1_name, m2_name = name_right, name_left
    else:
        m1_out, m2_out = ans_left, ans_right
        m1_name, m2_name = name_left, name_right

    prompt = JUDGE_PAIR_TEMPLATE.format(
        patient_prompt=case["patient_prompt"],
        gold_output=case["dataset_gold_output"],
        model_1_output=m1_out,
        model_2_output=m2_out,
    )
    reply = call_judge_llm(prompt)
    winner_label = (reply.get("winner") or "Tie").strip()
    if winner_label == "Model 1":
        real_winner = m1_name
    elif winner_label == "Model 2":
        real_winner = m2_name
    else:
        real_winner = "Tie"

    s1 = score_total(reply.get("model_1_scores", {}))
    s2 = score_total(reply.get("model_2_scores", {}))
    return {
        "reasoning": reply.get("reasoning"),
        "winner": real_winner,
        f"{name_left}_score": s2 if swap else s1,
        f"{name_right}_score": s1 if swap else s2,
    }


def judge_all_models_case(case, model_ids: list[str], field_map: dict, rng: random.Random):
    n = len(model_ids)
    answers = [(mid, case[resolve_field(field_map, case, mid)]) for mid in model_ids]
    perm = list(range(n))
    rng.shuffle(perm)
    shuffled = [answers[i] for i in perm]
    slot_to_id = {slot: shuffled[slot][0] for slot in range(n)}
    outputs = [shuffled[slot][1] for slot in range(n)]

    prompt = build_all_judge_prompt(
        n, case["patient_prompt"], case["dataset_gold_output"], outputs
    )
    reply = call_judge_llm(prompt)

    scores_by_id = {}
    raw_scores = reply.get("scores", {})
    for slot in range(1, n + 1):
        key = f"Model {slot}"
        mid = slot_to_id[slot - 1]
        sc = raw_scores.get(key, raw_scores.get(str(slot), {}))
        scores_by_id[mid] = score_total(sc) if isinstance(sc, dict) else float(sc or 0)

    ranking_ids = []
    for label in reply.get("ranking", []):
        idx = parse_model_label(label)
        if idx and 1 <= idx <= n:
            ranking_ids.append(slot_to_id[idx - 1])

    best_id = None
    best_label = reply.get("best", "")
    bidx = parse_model_label(best_label)
    if bidx and 1 <= bidx <= n:
        best_id = slot_to_id[bidx - 1]

    return {
        "reasoning": reply.get("reasoning"),
        "slot_to_model_id": {f"Model {k+1}": slot_to_id[k] for k in range(n)},
        "scores_by_model": scores_by_id,
        "ranking": ranking_ids,
        "best": best_id,
    }


def run_all(match_data, model_ids, field_map, label_map, out_path: Path, seed: int):
    rng = random.Random(seed)
    win_counts = {mid: 0 for mid in model_ids}
    score_sums = {mid: 0.0 for mid in model_ids}
    rank_points = {mid: 0 for mid in model_ids}
    judged = []

    for case in tqdm(match_data, desc="6模型同题盲审"):
        try:
            analysis = judge_all_models_case(case, model_ids, field_map, rng)
            for mid, sc in analysis["scores_by_model"].items():
                score_sums[mid] += sc
            for rank, mid in enumerate(analysis["ranking"]):
                rank_points[mid] += len(model_ids) - rank
            if analysis["best"]:
                win_counts[analysis["best"]] += 1
            judged.append({**case, "judge_all": analysis})
            time.sleep(0.5)
        except Exception as e:
            print(f"  样本 {case.get('id')} 失败: {e}")

    n = len(judged)
    report = {
        "mode": "all",
        "models": [{"id": mid, "label": label_map.get(mid, mid)} for mid in model_ids],
        "total_cases": n,
        "avg_scores": {mid: score_sums[mid] / n if n else 0 for mid in model_ids},
        "rank_points": rank_points,
        "best_model_counts": win_counts,
        "cases": judged,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 6 模型同题盲审汇总 ===")
    for mid in model_ids:
        print(
            f"  {label_map.get(mid, mid):40s}  均分={report['avg_scores'][mid]:.2f}  "
            f"最佳次数={win_counts[mid]}  排名积分={rank_points[mid]}"
        )
    print(f"  报告: {out_path}\n")
    return report


def run_pair(match_data, left_id, right_id, field_map, label_map, out_path: Path, seed: int):
    rng = random.Random(seed)
    sample = match_data[0]
    lf = resolve_field(field_map, sample, left_id)
    rf = resolve_field(field_map, sample, right_id)
    stats = {left_id: 0, right_id: 0, "Tie": 0}
    scores_l, scores_r = [], []
    judged = []

    for case in tqdm(match_data, desc=f"{left_id} vs {right_id}"):
        swap = rng.choice([True, False])
        try:
            analysis = judge_one_pair(
                case, case[lf], case[rf], left_id, right_id, swap
            )
            w = analysis["winner"]
            stats[w if w in stats else "Tie"] += 1
            scores_l.append(analysis[f"{left_id}_score"])
            scores_r.append(analysis[f"{right_id}_score"])
            judged.append({**case, "judge_pair": [left_id, right_id], "judge_analysis": analysis})
            time.sleep(0.5)
        except Exception as e:
            print(f"  样本 {case.get('id')} 失败: {e}")

    n = len(scores_l)
    report = {
        "pair": [left_id, right_id],
        "total_cases": n,
        f"{left_id}_avg_score": sum(scores_l) / n if n else 0,
        f"{right_id}_avg_score": sum(scores_r) / n if n else 0,
        "win_counts": stats,
        "cases": judged,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = len(match_data) - n
    print(f"\n=== {left_id} vs {right_id} ===")
    print(f"  成功: {n}/{len(match_data)}", end="")
    if failed:
        print(f"  失败: {failed}")
    else:
        print()
    if n:
        print(
            f"  {left_id} 均分: {report[f'{left_id}_avg_score']:.2f}  "
            f"{right_id} 均分: {report[f'{right_id}_avg_score']:.2f}"
        )
        print(f"  胜场: {stats}")
    print(f"  报告: {out_path}\n")
    return report


def majority_winner(votes: list[str], left_id: str, right_id: str) -> str:
    counts: dict[str, int] = {left_id: 0, right_id: 0, "Tie": 0}
    for v in votes:
        counts[v if v in counts else "Tie"] += 1
    best = max(counts.items(), key=lambda x: x[1])
    tied = [k for k, c in counts.items() if c == best[1]]
    if len(tied) > 1:
        return "Tie"
    return best[0]


def run_pair_repeat(
    match_data,
    left_id: str,
    right_id: str,
    field_map: dict,
    label_map: dict,
    out_path: Path,
    seed: int,
    repeats: int,
):
    sample = match_data[0]
    lf = resolve_field(field_map, sample, left_id)
    rf = resolve_field(field_map, sample, right_id)

    per_run_stats = []
    case_records = []
    majority_counts = {left_id: 0, right_id: 0, "Tie": 0}

    for case in tqdm(match_data, desc=f"{left_id} vs {right_id} x{repeats}"):
        run_results = []
        for run_idx in range(repeats):
            run_seed = seed + run_idx * 9973
            rng = random.Random(run_seed)
            swap = rng.choice([True, False])
            try:
                analysis = judge_one_pair(
                    case, case[lf], case[rf], left_id, right_id, swap
                )
                run_results.append(
                    {
                        "run": run_idx + 1,
                        "seed": run_seed,
                        "swap": swap,
                        "winner": analysis["winner"],
                        f"{left_id}_score": analysis[f"{left_id}_score"],
                        f"{right_id}_score": analysis[f"{right_id}_score"],
                        "reasoning": analysis.get("reasoning"),
                    }
                )
                time.sleep(0.5)
            except Exception as e:
                run_results.append(
                    {"run": run_idx + 1, "seed": run_seed, "error": str(e)}
                )

        winners = [r["winner"] for r in run_results if "winner" in r]
        if not winners:
            continue

        maj = majority_winner(winners, left_id, right_id)
        majority_counts[maj if maj in majority_counts else "Tie"] += 1

        vote_summary = {left_id: 0, right_id: 0, "Tie": 0}
        for w in winners:
            vote_summary[w if w in vote_summary else "Tie"] += 1

        case_records.append(
            {
                "id": case.get("id"),
                "patient_prompt": case.get("patient_prompt"),
                f"winner_run1": winners[0] if len(winners) > 0 else None,
                f"winner_run2": winners[1] if len(winners) > 1 else None,
                f"winner_run3": winners[2] if len(winners) > 2 else None,
                "winners_by_run": winners,
                "vote_counts": vote_summary,
                "majority_winner": maj,
                "runs_detail": run_results,
            }
        )

    for run_idx in range(repeats):
        rc = {left_id: 0, right_id: 0, "Tie": 0}
        for rec in case_records:
            w = rec["winners_by_run"][run_idx] if run_idx < len(rec["winners_by_run"]) else None
            if w:
                rc[w if w in rc else "Tie"] += 1
        per_run_stats.append({"run": run_idx + 1, "seed": seed + run_idx * 9973, "win_counts": rc})

    report = {
        "pair": [left_id, right_id],
        "labels": {left_id: label_map.get(left_id), right_id: label_map.get(right_id)},
        "repeats": repeats,
        "total_cases": len(case_records),
        "per_run_win_counts": per_run_stats,
        "majority_win_counts": majority_counts,
        "cases": case_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {left_id} vs {right_id}（同批样例重复 {repeats} 次）===")
    for pr in per_run_stats:
        print(f"  第{pr['run']}次 seed={pr['seed']}: {pr['win_counts']}")
    print(f"  多数票汇总（按样例）: {majority_counts}")
    print(f"  报告: {out_path}\n")

    # 逐条简表
    hdr = "样例 id | " + " | ".join(f"第{i}次" for i in range(1, repeats + 1)) + " | 多数票"
    print(hdr)
    for rec in case_records[:20]:
        w = rec["winners_by_run"]
        parts = [(w[i] if i < len(w) else "-") for i in range(repeats)]
        row = f"  {rec['id']:4} | " + " | ".join(f"{p:18}" for p in parts) + f" | {rec['majority_winner']}"
        print(row)
    if len(case_records) > 20:
        print(f"  ... 共 {len(case_records)} 条，详见 JSON")

    return report


def main():
    global USE_JSON_RESPONSE_FORMAT
    args = parse_args()
    USE_JSON_RESPONSE_FORMAT = args.json_response_format
    cfg = load_config(args.config)
    field_map = id_to_field(cfg)
    label_map = {m["id"]: m["label"] for m in cfg["models"]}
    model_ids = [m["id"] for m in cfg["models"]]

    match_path = Path(args.match_file)
    if not match_path.is_file():
        print(f"缺少 {match_path}，请先 run_eval_all.py && build_match_results.py")
        return
    _, match_data = load_match_file(match_path)
    if args.max_cases:
        match_data = match_data[: args.max_cases]

    out_dir = Path(args.output_dir)

    if args.mode == "all":
        run_all(
            match_data,
            model_ids,
            field_map,
            label_map,
            out_dir / "judge_all_models.json",
            args.seed,
        )
        return

    if args.mode == "pair":
        if not args.left or not args.right:
            raise ValueError("pair 模式需 --left 和 --right")
        pairs = [(args.left, args.right)]
    else:
        pairs = [tuple(p) for p in cfg.get("judge_pairs", [])]

    summaries = []
    for left_id, right_id in pairs:
        suffix = f"_repeat{args.repeats}" if args.repeats > 1 else ""
        out_file = out_dir / f"judge_{left_id}_vs_{right_id}{suffix}.json"
        if args.repeats > 1:
            rep = run_pair_repeat(
                match_data,
                left_id,
                right_id,
                field_map,
                label_map,
                out_file,
                args.seed,
                args.repeats,
            )
            summaries.append(
                {
                    "pair": f"{left_id} vs {right_id}",
                    "repeats": args.repeats,
                    "per_run": rep["per_run_win_counts"],
                    "majority_win_counts": rep["majority_win_counts"],
                }
            )
        else:
            rep = run_pair(
                match_data,
                left_id,
                right_id,
                field_map,
                label_map,
                out_file,
                args.seed,
            )
            summaries.append({"pair": f"{left_id} vs {right_id}", "win_counts": rep["win_counts"]})

    (out_dir / "judge_summary_pairs.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
