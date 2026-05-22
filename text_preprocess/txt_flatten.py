# 1. 读取原文件内容
with open('flashed_medical_words.txt', 'r', encoding='utf-8') as f:
    line = f.read()

# 2. 按空格切分单词，并用换行符拼接
# split() 会自动处理连续的空格，确保不会留下空行
words = line.split()
result = '\n'.join(words)

# 3. 将结果写入新文件
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(result)

print("处理完成！新文件已保存为 output.txt")