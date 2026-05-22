import os
import math
from tqdm import tqdm
from smoothnlp.algorithm.phrase import extract_phrase
from transformers import AutoTokenizer


def load_sentences(file_path):
    """读取预处理好的短句文本"""
    print(f"📖 正在加载预处理文本: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"📊 成功加载 {len(sentences)} 行文本。")
    return sentences


def discover_candidates(sentences, top_k=20000, max_phrase_len=6, min_freq=50, batch_size=50000,
                        checkpoint_path="all_raw_candidates.txt"):
    """
    步骤一：利用 smoothnlp 分批次抽取候选短语，引入本地 Checkpoint 缓存机制
    """
    # 【智能判定】如果本地已经存在未过滤的候选词表，直接读取，免去重复计算
    if os.path.exists(checkpoint_path):
        print(f"♻️  发现本地已存在未过滤的原始候选词表: {checkpoint_path}，正在直接加载...")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            final_candidates = [line.strip() for line in f if line.strip()]
        print(f"✨ 成功从缓存中恢复了 {len(final_candidates)} 个原始候选词，直接跳过统计挖掘阶段！")
        return final_candidates

    print(f"🚀 未发现本地缓存，开始分批次运行 smoothnlp 统计学新词发现...")
    total_sentences = len(sentences)
    num_batches = math.ceil(total_sentences / batch_size)

    unique_candidates = set()

    with tqdm(total=total_sentences, desc="[1/2] 文本统计挖掘中", unit="行") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_sentences)
            batch_corpus = sentences[start_idx:end_idx]

            batch_phrases = extract_phrase(
                corpus=batch_corpus,
                top_k=top_k,
                min_n=2,
                max_n=max_phrase_len,
                min_freq=max(2, int(min_freq * (len(batch_corpus) / total_sentences)))
            )

            for word in batch_phrases:
                if isinstance(word, str):
                    unique_candidates.add(word)

            pbar.update(len(batch_corpus))

    final_candidates = list(unique_candidates)[:top_k]
    print(f"✨ 统计学挖掘完成，全局提炼出 {len(final_candidates)} 个去重后的医学候选词。")

    # 【核心新增】将未过滤的词表保存到本地，方便后面中断时直接恢复
    print(f"💾 正在保存原始候选词表至本地缓存: {checkpoint_path} ...")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for word in final_candidates:
            f.write(f"{word}\n")
    print(f"✅ 缓存保存成功。")

    return final_candidates


def filter_with_qwen(candidates, tokenizer_path):
    """
    步骤二：利用 Qwen2.5 Tokenizer 进行逆向筛查
    """
    print(f"🤖 正在加载 Qwen2.5 Tokenizer: {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    new_medical_words = []
    print("🔍 正在逆向对比 Qwen2.5 原始分词表...")

    for word in tqdm(candidates, desc="[2/2] Qwen词表逆向比对", unit="词"):
        if word.isdigit() or len(word) <= 1:
            continue

        tokens = tokenizer.tokenize(word)

        if len(tokens) > 1:
            new_medical_words.append({
                "word": word,
                "token_count": len(tokens),
                "qwen_splits": tokens
            })

    return new_medical_words


def save_results(new_words, output_path):
    """将最终提炼的新词保存为标准文本和便于观察的 TSV 格式"""
    new_words_sorted = sorted(new_words, key=lambda x: x['token_count'], reverse=True)

    vocab_output = output_path.replace(".tsv", "_pure_vocab.txt")
    with open(vocab_output, "w", encoding="utf-8") as f:
        for item in new_words_sorted:
            f.write(f"{item['word']}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("医学新词\tQwen原本切分碎片数\tQwen切分后的样子\n")
        for item in new_words_sorted:
            f.write(f"{item['word']}\t{item['token_count']}\t{item['qwen_splits']}\n")

    print(f"💾 成果保存成功！")
    print(f"📝 纯词表路径 (微调使用): {vocab_output}")
    print(f"📊 详细分析报告路径 (人工审核): {output_path}")


if __name__ == "__main__":
    # 配置路径
    CLEANED_TXT = "2——cleaned_corpus.txt"
    OUTPUT_REPORT = "extracted_medical_new_words.tsv"
    QWEN_MODEL = "../models/qwen2.5-1.5B-Instruct"

    # 💥 指定未过滤词表的本地缓存文件名
    RAW_CHECKPOINT = "all_raw_candidates.txt"

    # 1. 加载数据
    sentences = load_sentences(CLEANED_TXT)

    if len(sentences) > 0:
        # 2. 统计挖掘候选词（内部会自动判断是否读取 RAW_CHECKPOINT 缓存）
        candidates = discover_candidates(
            sentences,
            top_k=20000,
            max_phrase_len=6,
            min_freq=50,
            batch_size=50000,
            checkpoint_path=RAW_CHECKPOINT
        )

        # 3. Qwen 词表逆向过滤
        new_medical_words = filter_with_qwen(candidates, QWEN_MODEL)

        # 4. 保存结果
        save_results(new_medical_words, OUTPUT_REPORT)

        print(
            f"\n🎉 大功告成！从你的医学文本中，一共压榨出 {len(new_medical_words)} 个 Qwen2.5 词表中所没有的医学垂直领域新词。")
    else:
        print("❌ 错误：读取的文本为空，请检查输入文件。")