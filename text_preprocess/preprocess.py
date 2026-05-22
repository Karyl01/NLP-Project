import regex as re
from tqdm import tqdm
import os


def clean_and_segment_line(line):
    """
    针对单行医学文本的高级清洗与分句逻辑
    """
    # 1. 统一处理编码与空白符（将全角空格、制表符等全部转换为标准单空格）
    line = re.sub(r'\s+', ' ', line).strip()
    if not line:
        return []

    # 2. 移除医疗文本中常见的无意义杂质（如：HTML标签、网页链接、连续的非中英文字符、特殊符号）
    line = re.sub(r'<[^>]+>', '', line)  # 移除 HTML 标签
    line = re.sub(r'https?://\S+|www\.\S+', '', line)  # 移除 URL

    # 3. 核心：保留中文字符、高频医学单位/特殊符号（如 %, /, pH, mg）、中英文句尾标点
    # 允许保留数字和英文字母，因为“非小细胞肺癌EGFR突变”、“3mg/dl”中的英文字母也是词的一部分
    # 过滤掉其他无意义的乱码符号
    line = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\%\/\.\-\+\s，。！？、：；（）(),.!?:]', '', line)

    # 4. 统计新词发现不需要巨大的长句，而是需要结构清晰的“短句”。
    # 我们根据中英文常见的句尾标点符号，将长段落切分成短句列表。
    sentences = re.split(r'[。！？\!?\n\r]', line)

    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # 过滤掉过短的句子（比如切分出来只有一个字的残缺句）
        if len(sentence) >= 3:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def process_large_txt(input_path, output_path, chunk_size_mb=50):
    """
    流式读取超大文本，分块处理并实时写入，防止内存爆炸
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入的源文件: {input_path}")

    total_size = os.path.getsize(input_path)
    print(f"🔄 发现待处理文件: {input_path} (大小: {total_size / (1024 * 1024):.2f} MB)")
    print("🚀 开始专业预处理...")

    # 使用 tqdm 结合文件指针展示高精度进度条
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="清洗进度")

        # 逐行流式读取
        for line in infile:
            pbar.update(len(line.encode('utf-8')))  # 更新进度条

            # 执行清洗并分句
            sentences = clean_and_segment_line(line)

            # 将清洗后的干净短句逐行写入新文件
            if sentences:
                for sent in sentences:
                    outfile.write(sent + '\n')

        pbar.close()

    print(f"✨ 预处理完成！清洗后的数据已保存至: {output_path}")
    print(f"📊 干净语料总行数: {sum(1 for _ in open(output_path, 'r', encoding='utf-8'))}")


if __name__ == "__main__":
    # 配置你的输入输出路径
    INPUT_TXT = "medical_clean_corpus.txt"
    OUTPUT_TXT = "2——cleaned_corpus.txt"

    # 执行清洗
    process_large_txt(INPUT_TXT, OUTPUT_TXT)