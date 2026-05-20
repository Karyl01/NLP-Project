import json
import re
import os
from datasets import load_dataset

# 1. 明确指向本地的 train_encyclopedia.json 文件路径
encyclopedia_path = "../medical_datasets/medical/pretrain/train_encyclopedia.json"

if not os.path.exists(encyclopedia_path):
    print(f"错误：在路径 [{os.path.abspath(encyclopedia_path)}] 处没有找到本地 JSON 文件！")
    exit(1)

# 2. 一键直接加载本地 JSON 文件
print(f"正在从本地路径读取并加载预训练子集: {os.path.abspath(encyclopedia_path)}")
encyclopedia_ds = load_dataset("json", data_files={"train": encyclopedia_path}, split="train")
print("1. 本地数据加载完成...")


def remove_hierarchical_numbering_operator(text):
    """
    算子一：层级与各种各类数字/顿号序号擦除算子
     1、 2、 一、 二、 ① ② 1. 2. 1) 2) 1） 2）以及开头的各类横杠
    """
    # 匹配数字/中文数字/带圈数字 后面紧跟点、顿号、括号、冒号、横杠等组合
    pattern_advanced_nums = r'(?:^\s*|\s+)(?:\d+|[一二三四五六七八九十]+|[a-zA-Z]|[①②③④⑤⑥⑦⑧⑨⑩])[\.\s、\)）:：\-]+'
    text = re.sub(pattern_advanced_nums, ' ', text)
    return text


def remove_isolated_brackets_operator(text):
    """
    算子二：不必要内容与孤立/残余括号成对擦除算子
    完美干掉: （菌）、（野生）以及像文本末尾顽固单出的 '）' 或 ')'
    """
    # 先去除带内容的括号，如（菌）、（野生）
    pattern_brackets_content = r'[\(（][^）\)]*[\)）]'
    text = re.sub(pattern_brackets_content, ' ', text)

    # 再强制清除任何落单的、未配对成功的残余括号本身
    text = re.sub(r'[\(\)（）\[\]｛｝\{\}]', ' ', text)
    return text


def punctuation_normalization_operator(text):
    """
    算子三：特殊首尾噪声与残余干扰标点清洗算子
    去除句子首尾由于剥离序号后残留下来的顿号、逗号、各种冒号
    """
    return text.strip(".,，。、:-：  ")


def clean_spaces_operator(text):
    """
    算子四：多余空格压缩算子
    将清洗过程中由于替换而产生的连续多个空格或制表符，统一无缝合并为一个标准空格
    """
    return re.sub(r'\s+', ' ', text).strip()


# =========================================================================
# 流水线执行核心循环 (Execution Pipeline)
# =========================================================================

clean_lines = []

for item in encyclopedia_ds:
    text = item["text"].strip()
    if not text:
        continue

    # 按照常见的标点或换行将文本切碎成短句，便于算子按行精细化清洗
    sentences = re.split(r'[\n;\s]+', text)
    for sent in sentences:
        if not sent.strip():
            continue

        # 依次流式流经四大核心算子
        sent = remove_hierarchical_numbering_operator(sent)
        sent = remove_isolated_brackets_operator(sent)
        sent = punctuation_normalization_operator(sent)
        sent = clean_spaces_operator(sent)

        # 相当于 Data-Juicer 的 text_length_filter
        # 医疗专用词汇至少需要一定的长度支持，过滤掉清洗后剩下小于等于 8 个字的废话或序号残渣碎片
        if len(sent) > 8:
            clean_lines.append(sent)

# 将清洗后的文本固化为 BPE 最爱的单行纯文本格式
with open("./medical_clean_corpus.txt", "w", encoding="utf-8") as f:
    for line in clean_lines:
        f.write(line + "\n")

print(f"\n整合升级版清洗完成！已生成纯净医疗语料：./medical_clean_corpus.txt")
print(f"过滤后有效总行数: {len(clean_lines)} 行")
print(f"纯净样本前 5 行展示：")
for idx, line in enumerate(clean_lines[:5]):
    print(f"  [{idx + 1}] {line}")