import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_medical_vocabulary(model_path, medical_vocab_txt, output_dir):
    print("正在加载原始 Qwen2.5 Tokenizer 与模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")  # 重缩放权重在CPU完成即可

    print(f"正在读取你清洗好的医学词汇表: {medical_vocab_txt}...")
    with open(medical_vocab_txt, "r", encoding="utf-8") as f:
        # 读取每一行清洗后的医学词，并过滤掉空行
        new_tokens = [line.strip() for line in f if line.strip()]

    print(f"准备添加的新词数量: {len(new_tokens)} 个。")

    # 1. 扩充 Tokenizer 词表
    # add_tokens 会返回成功添加的新词数量（自动去重）
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"Tokenizer 扩充成功！实际新增了 {num_added_tokens} 个专属医学 Token。")

    # 2. 核心步骤：重缩放模型的 Embedding 层和 LM Head 维度
    old_vocab_size = len(tokenizer) - num_added_tokens
    print(f"当前原始词表大小: {old_vocab_size} -> 扩充后目标词表大小: {len(tokenizer)}")

    model.resize_token_embeddings(len(tokenizer))
    print("模型 Embedding 层与 LM Head 维度重缩放完成（新增维度已自动随机初始化）。")

    # 3. 保存扩充后的完整 Tokenizer 和模型配置（仅保存架构和重置后的权重，方便后续微调调用）
    print(f"正在将融合后的新模型资产保存至: {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("专属医学底模资产准备就绪！")


if __name__ == "__main__":
    # 配置你的路径
    ORIGINAL_QWEN = "../models/qwen2.5-1.5B-Instruct"  # 你的原始模型路径
    MY_CLEANED_VOCAB = "output.txt"  # 你刚刚清洗出来的词汇txt
    NEW_MEDICAL_MODEL_DIR = "../models/qwen2.5-1.5B-MedVocab"  # 融合后保存的新目录

    merge_medical_vocabulary(ORIGINAL_QWEN, MY_CLEANED_VOCAB, NEW_MEDICAL_MODEL_DIR)