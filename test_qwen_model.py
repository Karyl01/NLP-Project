import os
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ==========================================
# 1. 关键配置：请修改为您本地模型的文件夹路径
# ==========================================
local_model_path = "./models/qwen2.5-1.5B-Instruct"

print(f"正在从本地路径加载模型和 Tokenizer: {os.path.abspath(local_model_path)}")

try:
    # 2. 从本地文件夹加载配套的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

    # 3. 从本地文件夹加载模型权重并自动挂载到 GPU
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,  # 消费级显卡（3090/4090/TITAN）强烈推荐 bfloat16
        device_map="auto",           # 自动检测当前最空闲的 GPU
        local_files_only=True        # 严格限制：只从本地读取，绝不联网下载
    )
    print("本地模型及架构载入显存成功！")
except Exception as e:
    print(f"加载失败，请检查本地路径是否正确，或者文件夹内文件是否完整。错误信息:\n{e}")
    exit(1)

# 4. 构建测试用的医疗对话模板
messages = [
    {"role": "system", "content": "你是一位专业的医疗 AI 助手。"},
    {"role": "user", "content": "你好！请问‘半夏白术天麻汤’通常是由哪几味中药组成的？它的主治功效是什么？"}
]

# 使用 Qwen 官方标准的 ChatML 格式拼接 Prompt
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 5. 配置流式打印机 (Streamer) 产生逐字蹦出的打字机效果
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 将模型生成任务丢进后台线程，防止主线程因等待推理而卡死
generation_kwargs = dict(
    **model_inputs,
    streamer=streamer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

print("\n[Qwen 本地模型开始生成回答] -> \n")
thread.start()

# 6. 主线程负责从 Streamer 中流式读取 Token 并实时打印
for new_text in streamer:
    print(new_text, end="", flush=True)

print("\n\n测试完成！模型完全生效。")