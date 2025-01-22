from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    deepspeed,
)
from args_g import args_global
import json
from tqdm import tqdm
import random
from dataset import LLMFintuneDataset
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from trainer_utils import preprocess
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def set_train_finetune_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/my_qwen_finetuned_output",
        help="微调模型输出路径",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="微调训练的batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="num train epochs of model finetune",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="the learning rate of model finetune",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="model finetune save model steps",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=512,
        help="max_length of model finetune",
    )


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()

    trainer._save(output_dir, state_dict=state_dict)


def train_finetune(parser_parent: argparse.ArgumentParser):
    set_train_finetune_args(parser=parser_parent)
    args = parser_parent.parse_args()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name, trust_remote_code=True
    )

    train_datasets = LLMFintuneDataset(args=args, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name, trust_remote_code=True, device_map="auto"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_datasets,
        args=training_args,
    )

    if args.test:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=args.output_dir, bias=None
        )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=args.output_dir, bias=None
    )


if __name__ == "__main__":
    parser_parent = args_global()

    parser = argparse.ArgumentParser(parents=[parser_parent])

    parser.add_argument(
        "--train_finetune-exec-function",
        type=str,
        default="train_finetune",
        help="微调大模型后需要预测使用的模型的路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size of LLM finetune",
    )
    parser.add_argument(
        "--train_dataset_filepath",
        type=str,
        default="datas/generated_deepspeed_samples.json",
        help="train datasets file path of finetune",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="the input tokens num of LLM finetune",
    )

    parser.add_argument(
        "--use_lora",
        type=bool,
        default=False,
        help="is or not use lora",
    )
    parser.add_argument(
        "--should_save",
        type=bool,
        default=True,
        help="shoule have is always true",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="test function",
    )
    args = parser.parse_args()

    if args.train_finetune_exec_function == "train_finetune":
        train_finetune(parser_parent=parser)
