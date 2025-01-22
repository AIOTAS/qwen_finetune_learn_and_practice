from torch.utils.data import Dataset
import torch.nn as nn
import json
from args_g import args_global
import argparse
from trainer_utils import preprocess
from transformers import AutoTokenizer


class LLMFintuneDataset(Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.samples = json.load(
            open(args.train_dataset_filepath, "r", encoding="utf-8")
        )

        sources = [sample["conversations"] for sample in self.samples]

        input_output_datas_dict = preprocess(
            sources=sources, tokenizer=tokenizer, max_len=args.max_len
        )

        self.input_ids = input_output_datas_dict["input_ids"]
        self.labels = input_output_datas_dict["labels"]
        self.attention_mask = input_output_datas_dict["attention_mask"]

        assert len(self.input_ids) == len(self.labels)

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return dict(
            input_ids=self.input_ids[index],
            labels=self.labels[index],
            attention_mask=self.attention_mask[index],
        )


if __name__ == "__main__":
    parser = args_global()
    parser.add_argument(
        "--train_dataset_filepath",
        type=str,
        default="datas/generated_deepspeed_samples.json",
        help="finetune train dataset file path",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="max_length of model finetune",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name, trust_remote_code=True
    )

    train_finetune_datasets = LLMFintuneDataset(args=args, tokenizer=tokenizer)

    print(train_finetune_datasets[0])
