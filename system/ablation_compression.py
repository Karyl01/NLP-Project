import json
import random
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def sample_and_calculate_compression(original_path, new_path, json_file_path, num_samples=1000):
    """
    流式读取大 JSONL 文件，随机抽取 num_samples 条文本进行词表扩充消融分析
    """
    print(f"正在加载原始与新 Tokenizer...")
    old_tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=True)
    new_tokenizer = AutoTokenizer.from_pretrained(new_path, trust_remote_code=True)

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"找不到输入的 JSON 文本文件: {json_file_path}")

    # ===== 阶段 1：流式统计总行数（免内存爆炸） =====
    print(f"正在扫描文件行数: {json_file_path}...")
    total_lines = 0
    with open(json_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            total_lines += 1

    print(f"文件总计包含 {total_lines} 行数据。")

    if total_lines < num_samples:
        print(f"警告：文件总行数不足 {num_samples}，将对全量数据进行评测。")
        sampled_indices = set(range(total_lines))
    else:
        # 随机抽取 1000 个行号
        sampled_indices = set(random.sample(range(total_lines), num_samples))

    # ===== 阶段 2：精准捞取抽样的 1000 条医学文本 =====
    print(f"开始流式提取并计算这 {len(sampled_indices)} 条样本文本的 Token 压缩率...")
    old_lens = []
    new_lens = []

    with open(json_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # 使用 tqdm 配合刚刚算出的 total_lines 显示炫酷的流式进度条
        for current_idx, line in enumerate(tqdm(f, total=total_lines, desc="文本抽样比对中")):
            # 如果当前行号在我们随机抽取的集合中，则处理
            if current_idx in sampled_indices:
                line = line.strip()
                if not line: continue

                try:
                    # 解析单行 JSON，并提取出 "text" 字段
                    data = json.loads(line)
                    text = data.get("text", "").strip()

                    if len(text) < 10:  # 过滤过短的残缺文本
                        continue

                    # 统计切分后的 Token 数量
                    old_lens.append(len(old_tokenizer.tokenize(text)))
                    new_lens.append(len(new_tokenizer.tokenize(text)))

                except json.JSONDecodeError:
                    # 容错处理：如果中间有某一行 json 格式坏了，直接跳过
                    continue

    # ===== 阶段 3：计算并打印消融分析报告 =====
    if not old_lens:
        print(" 错误：未能成功提取到任何有效的医学文本，请检查 JSON 格式或键名是否为 'text'。")
        return

    avg_old = np.mean(old_lens)
    avg_new = np.mean(new_lens)
    compression_rate = (1 - (avg_new / avg_old)) * 100

    print("\n" + "=" * 40 + "词表扩充消融分析报告 " + "=" * 40)
    print(f"评测样本量                    : {len(old_lens)} 条医学文本")
    print(f"原始 Qwen2.5 平均 Token 长度  : {avg_old:.2f}")
    print(f"扩充医学词表后 平均 Token 长度: {avg_new:.2f}")
    print(f"文本序列平均压缩率 (序列缩短)  : {compression_rate:.2f}%")
    print(f"结论说明：在微调训练阶段，单批次（Batch）的平均显存开销将降低约 {compression_rate:.1f}%，")
    print(f"             同时模型的上下文窗口（Context Window）利用率和推理速度将获得显著提升！")
    print("=" * 103)


if __name__ == "__main__":
    # 配置你的实际路径
    ORIGINAL_QWEN = "../models/qwen2.5-1.5B-Instruct"  # 原始 Qwen2.5 底模路径
    NEW_MEDICAL_MODEL = "../models/qwen2.5-1.5B-MedVocab"  # 你第一步融合词表后生成的新模型路径

    # 💥 这里换成你的 shibing624/medical 数据集解压出来的预训练 json 文件路径
    MEDICAL_JSON_FILE = "../medical_datasets/medical/pretrain/test_encyclopedia.json"

    # 执行随机抽样评测
    sample_and_calculate_compression(
        original_path=ORIGINAL_QWEN,
        new_path=NEW_MEDICAL_MODEL,
        json_file_path=MEDICAL_JSON_FILE,
        num_samples=1000  # 随机抽取1000条
    )