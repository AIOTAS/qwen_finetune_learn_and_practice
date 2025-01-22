import torch
from torch.utils.data import Dataset
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, Trainer, TrainingArguments
import json
from transformers.trainer_pt_utils import LabelSmoother
import argparse
from args_g import args_global

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "你是一个擅长于回答知识的助手。",
):
    tokenizer.pad_token_id = tokenizer.eod_id

    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids

    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    input_ids, targets = [], []

    for i, source in enumerate(sources):
        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += (
            [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )

        assert len(input_id) == len(target)

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer(sentence["value"]).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id

            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )

            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids))
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)

        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    print(f"{len(input_ids)=}")

    print(f"{tokenizer.decode(input_ids[0])=}")

    for id in targets[0]:
        str_ = tokenizer.decode(id) if id != -100 else "[PAD]"
        print(str_.replace("\n", "\\n"), end="")

    print()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class TrainerDataset(Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        data_array = json.load(open(args.sample_json_filepath, "r", encoding="utf-8"))
        print(len(data_array))
        exit()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_json = json.loads(self.x[index])
        print(data_json)
        inputs = self.tokenizer(
            data_json["question"], return_tensors="pt", padding=True
        )
        return dict(
            input_ids=inputs["input_ids"],
        )


class TrainerRNNModule(nn.Module):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()

        self.embedding = nn.Embedding(tokenizer.vocab_size, 768)

        self.lstm = nn.LSTM(
            input_size=768, hidden_size=512, num_layers=2, bidirectional=True
        )

        self.linear = nn.Sequential(
            nn.Linear(2 * 512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, tokenizer.vocab_size),
        )

    def forward(self, x):
        print(x)
        x = self.embedding(x)

        x, (hidden, c) = self.lstm(x)

        out = self.linear(x)

        return out


def finetune(args):
    tokenizer = AutoTokenizer.from_pretrained(
        "modules/Qwen-Qwen-1_8B-Chat", trust_remote_code=True
    )

    tokenizer.pad_token_id = tokenizer.eod_id

    train_datasets = TrainerDataset(args=args, tokenizer=tokenizer)

    model = TrainerRNNModule(tokenizer=tokenizer)

    print(model)

    training_args = TrainingArguments(
        output_dir="./models/output_trainer",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        save_steps=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_datasets)

    trainer.train()


if __name__ == "__main__":
    parser = args_global()
    parser.add_argument(
        "--train-demo-execute-function",
        type=str,
        required=True,
        help="train demo.py 选择执行的函数",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "modules/Qwen-Qwen-1_8B-Chat", trust_remote_code=True
    )

    tokenizer.pad_token_id = tokenizer.eod_id

    if args.train_demo_execute_function == "preprocess":
        sources = [
            [
                {"from": "user", "value": "你是谁？"},
                {"from": "assistant", "value": "我是双天至尊AIOTAS的助手"},
            ]
        ]
        preprocess(sources=sources, tokenizer=tokenizer, max_len=512)
    elif args.train_demo_execute_function == "finetune":
        finetune(args=args)
