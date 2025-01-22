import argparse


def args_global():
    parser = argparse.ArgumentParser(description="大模型微调实战", add_help=False)
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="Qwen/Qwen-1_8B-Chat",
        help="llm 大模型名称或者路径",
    )
    parser.add_argument(
        "--generate_samples_num",
        type=int,
        default=10000,
        help="数据增广的数量",
    )
    parser.add_argument(
        "--sample_json_filepath",
        type=str,
        default="datas/generated_deepspeed_samples.json",
        help="sample json 文件路径",
    )
    parser.add_argument(
        "--finetune_run_machine_type",
        type=str,
        default="cuda",
        help="微调大模型运行在GPU还是CPU",
    )
    parser.add_argument(
        "--finetune-model-save-dir",
        type=str,
        default="models/",
        help="微调大模型后模型保存目录",
    )
    parser.add_argument(
        "--finetune-prompt-template-filepath",
        type=str,
        default="prompt_template.txt",
        help="微调大模型后模型保存目录",
    )

    parser.add_argument(
        "--finetuned_predict_model_path",
        type=str,
        default="models/deepspeed_fintuned_model",
        help="微调大模型后需要预测使用的模型的路径",
    )
    return parser
