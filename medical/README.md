---
license: apache-2.0
language:
- zh
tags:
- text-generation
pretty_name: medical
task_categories:
- text-generation
size_categories:
- n<1K
---

# Dataset Card for medical
中文医疗数据集

- LLM Supervised Finetuning repository: https://github.com/shibing624/textgen
- MeidcalGPT repository: https://github.com/shibing624/MedicalGPT
  
## Dataset Description

medical is a Chinese Medical dataset. 医疗数据集，可用于医疗领域大模型训练。

```
tree medical
|-- finetune  # 监督微调数据集，可用于SFT和RLHF
|   |-- test_en_1.json
|   |-- test_zh_0.json
|   |-- train_en_1.json
|   |-- train_zh_0.json
|   |-- valid_en_1.json
|   `-- valid_zh_0.json
|-- medical.py # hf dataset 数据展示用
|-- pretrain # 二次预训练数据集
|   |-- medical_book_zh.json
|   |-- test_encyclopedia.json
|   |-- train_encyclopedia.json
|   `-- valid_encyclopedia.json
|-- README.md
`-- reward # 奖励模型数据集
    |-- test.json
    |-- train.json
    `-- valid.json
```




### Original Dataset Summary

#### pretrain
- train_encyclopedia.json: 共36万条，来自医疗百科数据[FreedomIntelligence/huatuo_encyclopedia_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa) , 拼接 questions 和 answers，形成 text 文本字段，语句通顺，用于预训练注入医疗知识。
- medical_book_zh.json: 共8475条，来自医疗教材的文本数据，来源：https://github.com/jind11/MedQA， 原始数据集：[google drive](https://drive.google.com/u/0/uc?export=download&confirm=t&id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw) ，只对长段落切分为2048字的小段落了。
#### finetune
- train_zh_0.json: 共195万条，来自1）中文医疗对话数据集[Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)的六个科室医疗问诊数据，
有79万条；2）在线医疗百科 huatuo_encyclopedia_qa ，有36万条；3）医疗知识图谱 huatuo_knowledge_graph_qa，有79万条。三部分合并，共195万条。
- train_en_1.json：共11万条，来自英文医疗问诊对话数据[Kent0n-Li/ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)，合并了HealthCareMagic-100k、GenMedGPT-5k 数据集，共11万条。
#### reward
- train.json 共4000条，问题来自中文医疗对话数据集[Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)的随机4000条提问，`response_chosen`来自该数据集的医生答复，
`response_rejected`来自本草模型[SCIR-HI/Huatuo-Llama-Med-Chinese](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)的答复。

### Supported Tasks and Leaderboards
中文医疗对话模型

The dataset designed for medical task training pretrained language models.

### Languages

The data are in Chinese.

## Dataset Structure

### Data Instances

An example of "train" looks as follows:

head pretrain/train_encyclopedia.json
```json
{"text": "怀孕后嘴巴很淡怎么办？有孕妇在怀孕之后，发现自己嘴巴比较淡，出现这种情况的原因其实也非常的复杂，首先和妊娠反应有直接的关系，这是一种正常的情况，另外有些孕妇平常不注意口腔卫生，舌苔比较厚或者自己有了一些消化系统方面的疾病，这就要求人们必须要及时的进行处理。女性在怀孕之后，身体就会出现一些明显的变化，首先人们月经会停止，另外也会有恶心、呕吐等一些妊娠反应，不过这些都是正常的。有些孕妇发现自己在怀孕之后，口味发生了很大的变化，嘴巴变得非常的淡。其实这也和激素变化有直接的关系，可能是妊娠反应所致，在怀孕期间，因为受到体内激素水平的变化，所以就会有肠胃系统的改变，人们可能会出现食欲不振，消化不良等症状表现，也有一些孕妇会发现自己嘴巴没有味道，会有口苦的症状，而这也是正常的孕期反应，人们在平常要多喝一些水，多吃一些清淡营养的食物慢慢就会改善。也有可能是舌苔过厚所致，孕妇嘴巴里面没有味道，很有可能是舌苔比较重、舌苔过厚导致的，这样就会影响到味蕾对味道的敏感度，不仅嘴巴里面没有味道，甚至有些孕妇在说话的时候也会有口臭，这就要求人们在每天早上漱口的时候，必须要用牙刷刷舌苔开始，不要清理的特别深，以免人们会有呕吐，慢慢习惯之后再往深一点的清洗，一般2到3天就会得到改善。嘴巴感到比较淡，其实也和脾胃虚寒有直接的关系，消化系统疾病，内分泌疾病，营养不良等，但有可能导致舌头部位因为味蕾的敏感度下降，产生口淡之感，患者会有食欲不振的表现，发现病症及时就诊治疗。"}
```

head finetune/train_zh_0.json
```json
{"instruction": "曲匹地尔片的用法用量", "input": "", "output": "注意：同种药品可由于不同的包装规格有不同的用法或用量。本文只供参考。如果不确定，请参看药品随带的说明书或向医生询问。口服。一次50～100mg（1-2片），3次/日，或遵医嘱。"}
```

head finetune/train_en_1.json
```json
{"instruction": "If you are a doctor, please answer the medical questions based on the patient's description.", "input": "Doctor, I have been experiencing sudden and frequent panic attacks. I don't know what to do.", "output": "Well, based on what you're telling me, it sounds like you may be suffering from panic disorder. The best course of action is to start with psychotherapy and mental health counseling. Additionally, we should conduct an electrocardiogram to make sure that there are no physical issues causing your panic attacks. We will also need to perform a depression screen and a toxicology screen to rule out any other underlying causes. Finally, I would recommend a comprehensive psychological and psychiatric evaluation and therapy to help manage your symptoms."}
```

head reward/train.json
```json
{"question": "孩子感统失调，学习不行怎么办？，孩子说话晚，走路晚，现在小学二年级，学习跟不上，理解力差，运动协调性差，家里很着急，不知怎么办。", "response_chosen": "病情分析：你好!孩子说话晚，走路也晚，很可能是大脑本身发育不好引起的发育迟缓。而五岁时所致的智力检查为临界范围，那就是说孩子的智商是有问题的，也应考虑与大脑发育不好有关。指导意见：人的大脑在头一年发育最快，可塑性最强，在头三年可塑性还是可以的，超过三岁再进行训练，效果就不怎么好了。建议再给孩子做一做智力测试，如果孩子的智商还是在临界范围，那就要考虑让孩子去特殊学校进行康复训练，而不是继续在普通小学就读，否则对孩子来说，就是强人所难了。希望自己的孩子能聪明，这是每个家长都会有的心愿，但如果孩子自身的条件就是不能跟上同龄孩子，那家长也要面对这个事实的，对吗？医生询问：", "response_rejected": "建议家长先带孩子去正规医院做全面检查以确定病因和病情严重程度；同时可以进行物理治疗、康复训练等辅助治疗方法。"}
```

### Data Fields

#### 预训练数据集 pretrain
字段解释：
- text: 文本

#### 指令微调数据集 finetune
字段解释：
- instruction: 指令
- input：问题（可为空）
- output：答复

#### 奖励模型数据集 reward
字段解释：
- question: 问题
- response_chosen: 优质回答
- response_rejected: 低质回答 
  
### Data Splits

```
> wc -l medical/*/*
       500 medical/finetune/test_en_1.json
       500 medical/finetune/test_zh_0.json
    116617 medical/finetune/train_en_1.json
   1949972 medical/finetune/train_zh_0.json
       500 medical/finetune/valid_en_1.json
       500 medical/finetune/valid_zh_0.json
      8475 medical/pretrain/medical_book_zh.json
       500 medical/pretrain/test_encyclopedia.json
    361420 medical/pretrain/train_encyclopedia.json
       500 medical/pretrain/valid_encyclopedia.json
       100 medical/reward/test.json
      3800 medical/reward/train.json
       100 medical/reward/valid.json
   2443484 total
```

### Licensing Information

The dataset is available under the Apache 2.0.


### Citation Information

- https://github.com/Toyhom/Chinese-medical-dialogue-data
- https://github.com/FreedomIntelligence/Huatuo-26M/blob/main/README_zh-CN.md
- https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa
- https://huggingface.co/datasets/FreedomIntelligence/huatuo_knowledge_graph_qa
- https://github.com/Kent0n-Li/ChatDoctor

附上几个优质的reward model dataset: 
- https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise
- https://huggingface.co/datasets/sunzeyeah/chinese_chatgpt_corpus
- https://huggingface.co/datasets/Cohere/miracl-zh-queries-22-12
- https://huggingface.co/datasets/Dahoas/rm-static
  
### Contributions

[shibing624](https://github.com/shibing624) 整理并上传