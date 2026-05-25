# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Natural Language Generation Chinese Corpus.(medical)
"""

import os
import json
import medical_datasets
_DESCRIPTION = """纯文本数据，中文医疗数据集，包含预训练数据的百科数据，指令微调数据和奖励模型数据。"""
_HOMEPAGE = "https://github.com/shibing624/MedicalGPT"
_CITATION = ""
_LICENSE = ""
_BASE_URL = "https://huggingface.co/datasets/shibing624/medical/resolve/main/"
# file url: https://huggingface.co/datasets/shibing624/medical/resolve/main/finetune/test_zh_0.json

class NewDataset(medical_datasets.GeneratorBasedBuilder):
    """Medical Chinese Version"""

    VERSION = medical_datasets.Version("1.0.1")

    BUILDER_CONFIGS = [
        medical_datasets.BuilderConfig(name="pretrain", version=VERSION, description="pretrain data"),
        medical_datasets.BuilderConfig(name="finetune", version=VERSION, description="finetune data"),
        medical_datasets.BuilderConfig(name="reward", version=VERSION, description="reward data"),
    ]

    def _info(self):
        if self.config.name == "pretrain":
            features = medical_datasets.Features(
                {
                    "text": medical_datasets.Value("string")
                }
            )
        elif self.config.name == 'finetune': 
            features = medical_datasets.Features(
                {
                    "instruction": medical_datasets.Value("string"),
                    "input": medical_datasets.Value("string"),
                    "output": medical_datasets.Value("string")
                }
            )
        elif self.config.name == 'reward': 
            features = medical_datasets.Features(
                {
                    "question": medical_datasets.Value("string"),
                    "response_chosen": medical_datasets.Value("string"),
                    "response_rejected": medical_datasets.Value("string")
                }
            )
        
        return medical_datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_url = _BASE_URL + self.config.name

        if self.config.name == 'pretrain':
            return [
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/train_encyclopedia.json"),
                        "split": "train"
                    },
                ),
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/valid_encyclopedia.json"),
                        "split": "dev"
                    },
                ),
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/test_encyclopedia.json"),
                        "split": "test"
                    },
                ),
            ]
        elif self.config.name == 'finetune':
            return [
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract([f"{data_url}/train_zh_0.json", f"{data_url}/train_en_1.json"]),
                        "split": "train"
                    },
                ),
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract([f"{data_url}/valid_zh_0.json", f"{data_url}/valid_en_1.json"]),
                        "split": "dev"
                    },
                ),
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract([f"{data_url}/test_zh_0.json", f"{data_url}/test_en_1.json"]),
                        "split": "test"
                    },
                ),
            ]
        elif self.config.name == 'reward':
            return [
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/train.json"),
                        "split": "train"
                    },
                ),
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/valid.json"),
                        "split": "dev"
                    },
                ),
                medical_datasets.SplitGenerator(
                    name=medical_datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/test.json"),
                        "split": "test"
                    },
                ),
            ]
        
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        id = 0
        if isinstance(filepath, str):
            filepath = [filepath]
        for file in filepath:
            with open(file, encoding="utf-8") as f:
                for key, row in enumerate(f):
                    data = json.loads(row)
                    if self.config.name == "pretrain":
                        yield id, {
                            "text": data["text"]
                        }
                    elif self.config.name == 'finetune':
                        yield id, {
                            "instruction": data["instruction"],
                            "input": data["input"],
                            "output": data["output"]
                        }
                    elif self.config.name == 'reward':
                        yield id, {
                            "question": data["question"],
                            "response_chosen": data["response_chosen"],
                            "response_rejected": data["response_rejected"]
                        }
                    id += 1


