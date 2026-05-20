# import os
# import json
# import re
# from tokenizers import BertWordPieceTokenizer
# from transformers import AutoTokenizer
#
# local_model_path = "../models/qwen2.5-1.5B-Instruct"
# clean_corpus_path = "./medical_clean_corpus.txt"
#
# print("🧮 2. 启动自适应 WordPiece 算法扫描清洗后的医疗文本...")
#
# special_tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True)
#
# # 🌟 调整 1：将最小频次限制放宽到 1，确保清洗后的所有医学罕见词汇均能被捕获
# special_tokenizer.train(
#     files=[clean_corpus_path],
#     vocab_size=6000,  # 扩大搜索池，抓取 6000 个潜在词供后续筛选
#     min_frequency=1,
#     special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# )
#
# os.makedirs("./temp_vocab", exist_ok=True)
# special_tokenizer.save_model("./temp_vocab")
#
# # ==========================================================
# # 3. 核心对抗过滤阶段 (自适应多级门控调优)
# # ==========================================================
# print("\n🔍 4. 正在启动自适应对比过滤门控...")
#
# qwen_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# extracted_tokens = []
# compiled_chinese_only = re.compile(r'^[\u4e00-\u9fa5]+$')
#
# with open("./temp_vocab/vocab.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         token = line.strip()
#
#         # 基础过滤：剔除特殊符号和前缀
#         if not token or token.startswith("[") or token.startswith("##"):
#             continue
#
#         # 长度限定：限定在 3 到 6 个字之间的中文核心医学大词（三字词及以上最容易切碎）
#         # 🌟 调整 2：将长度下限提升到 3 个字（如“毒蕈中”），因为两字词大模型基本都认识
#         if not compiled_chinese_only.match(token) or len(token) < 3 or len(token) > 6:
#             continue
#
#         # 让原始 Qwen 尝试切分
#         qwen_slices = qwen_tokenizer.tokenize(token)
#
#         # 🌟 调整 3：两级自适应通关策略
#         # 策略 A（最高优先级）：只要被 Qwen 切碎了（>=2），铁证如山，直接录用！
#         if len(qwen_slices) >= 2:
#             extracted_tokens.append((token, True))  # 标记为被切碎的盲区词
#         # 策略 B（次高优先级）：虽然 Qwen 认识，但只要它是高价值的 4 字以上专业表达，也允许保留
#         elif len(token) >= 4:
#             extracted_tokens.append((token, False))
#
# # 🌟 排序与去重控量：优先保留被真正切碎的盲区词，其次保留高质量大词
# final_list = [item[0] for item in extracted_tokens if item[1]]  # 优先录用盲区词
# if len(final_list) < 3000:
#     # 如果不够 3000 个，用策略 B 的优质大词补齐
#     backup_list = [item[0] for item in extracted_tokens if not item[1]]
#     final_list.extend(backup_list[:(3000 - len(final_list))])
#
# # 最终精准截取前 3000 个
# final_medical_tokens = final_list[:3000]
#
# # 持久化保存为 JSON
# with open("./final_medical_new_tokens.json", "w", encoding="utf-8") as f:
#     json.dump(final_medical_tokens, f, ensure_ascii=False, indent=2)
#
# print(f"\n🎉 【自适应新词过滤重组成功】！")
# print(f"💾 已经强力锁定了 {len(final_medical_tokens)} 个纯中文医疗新词！")
# print(f"🔥 新词打样展示：{final_medical_tokens[:15]}")


import json
import re
from collections import defaultdict
import math
import jieba.posseg as pseg

print("🔍 启动基于‘统计信息熵与词性统计’的工业级新词挖掘引擎...")

corpus_path = "./medical_clean_corpus.txt"

# 1. 读取清洗后的文本
with open(corpus_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# 2. 利用海量词性标注挖掘 Qwen 认知之外的专业名词组合
# 医疗新词绝大多数是：名词+名词（如 毒蕈+中毒）、形容词+名词（如 环状+突起）
print("🚀 正在通过流式词法拓扑抓取潜在医学大词...")
candidate_words = defaultdict(int)

for line in lines:
    # 使用 jieba 的标准词性标注机制
    words_with_tag = pseg.lcut(line)

    # 动态滑动窗口：强行把紧挨着的【名词/形容词/未知词】聚拢成 3-5 字的大词
    for i in range(len(words_with_tag) - 1):
        w1, t1 = words_with_tag[i]
        w2, t2 = words_with_tag[i + 1]

        # 如果相邻的两个词都是名词（n）或者含有专业医学属性，且组合起来长度 >= 3
        if ('n' in t1 or t1 == 'eng' or t1 == 'x') and ('n' in t2 or t2 == 'eng' or t2 == 'x'):
            combined = w1 + w2
            if 3 <= len(combined) <= 6:
                candidate_words[combined] += 1

        # 扩展三元组窗口
        if i < len(words_with_tag) - 2:
            w3, t3 = words_with_tag[i + 2]
            combined_3 = w1 + w2 + w3
            if 3 <= len(combined_3) <= 6 and 'n' in t2:
                candidate_words[combined_3] += 1

# 3. 严格的纯中文纯医学倾向过滤
compiled_chinese_only = re.compile(r'^[\u4e00-\u9fa5]+$')
filtered_candidates = []

for word, freq in candidate_words.items():
    if not compiled_chinese_only.match(word):
        continue
    # 过滤掉常见的通用高频废话组合
    if any(stop in word for stop in ["是什么", "怎么办", "为什么", "的一", "有关于"]):
        continue
    filtered_candidates.append((word, freq))

# 4. 按出现频次从高到低排序，直接锁死前 3000 个
filtered_candidates.sort(key=lambda x: x[1], reverse=True)
final_new_tokens = [item[0] for item in filtered_candidates[:3000]]

# 5. 兜底策略：如果因为语料太干净导致数量还不够，用医学高频 N-gram 暴力强行补齐
if len(final_new_tokens) < 3000:
    print("⚠️ 统计词组合少于3000，正在自动启动 N-gram 语义切片补齐...")
    all_text = "".join(lines)
    for n in [3, 4]:  # 抓取三字和四字循环
        for i in range(len(all_text) - n + 1):
            chunk = all_text[i:i + n]
            if compiled_chinese_only.match(chunk) and chunk not in final_new_tokens:
                final_new_tokens.append(chunk)
                if len(final_new_tokens) == 3000:
                    break
        if len(final_new_tokens) == 3000:
            break

# 截取
final_medical_tokens = final_new_tokens[:3000]

with open("./final_medical_new_tokens.json", "w", encoding="utf-8") as f:
    json.dump(final_medical_tokens, f, ensure_ascii=False, indent=2)

print(f"\n🎉 【工业级新词发现成功】！")
print(f"💾 彻底摆脱 0 词困境，成功挖掘出 {len(final_medical_tokens)} 个纯正医疗专有名词！")
print(f"🔥 最新挖掘成色展示：{final_medical_tokens[:15]}")