import pandas as pd
import json
from tqdm import tqdm
from args_g import args_global
import random
import argparse


def data_dealing():
    data_df = pd.read_excel("datas/最全中国各省份城市编码以及经纬度数据.xlsx")
    print(data_df.head())
    city_list = data_df["省份城市"].tolist()
    json.dump(
        city_list,
        open("datas/city_names.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )


def qwen_official_finutune_dataset_prepreare(args):
    contain_year = True

    question_list = [
        ("{city}{year}年{month}月{day}日的天气", "{year}-{month}-{day}", contain_year),
        ("{city}{year}年{month}月{day}号的天气", "{year}-{month}-{day}", contain_year),
        ("{city}{month}月{day}日的天气", "{month}-{day}", not contain_year),
        ("{city}{month}月{day}号的天气", "{month}-{day}", not contain_year),
        ("{year}年{month}月{day}日{city}的天气", "{year}-{month}-{day}", contain_year),
        ("{year}年{month}月{day}号{city}的天气", "{year}-{month}-{day}", contain_year),
        ("{month}月{day}日{city}的天气", "{month}-{day}", not contain_year),
        ("{month}月{day}号{city}的天气", "{month}-{day}", not contain_year),
    ]

    city_names = json.load(open("datas/city_names.json", "r", encoding="utf_8"))

    with open(args.finetune_prompt_template_filepath, "r", encoding="utf_8_sig") as fr:
        prompt_template = fr.read()

    generated_samples = []

    count = 0

    first_conversation = {
        "id": str(count).zfill(8),
        "conversations": [
            {"from": "user", "value": "你好"},
            {
                "from": "assistant",
                "value": "我是双天至尊AIOTAS的助手,支持天气预报语句的信息提取。",
            },
        ],
    }

    generated_samples.append(first_conversation)

    print("start generate text samples.")

    for current_i in tqdm(range(args.generate_samples_num)):
        count += 1
        question = question_list[random.randint(0, len(question_list) - 1)]
        city = city_names[random.randint(0, len(city_names) - 1)]
        year = random.randint(1992, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 31)
        time_str = (
            question[1].format(year=year, month=month, day=day)
            if question[2]
            else question[1].format(month=month, day=day)
        )
        question = question[0].format(city=city, year=year, month=month, day=day)

        generated_samples.append(
            {
                "id": str(count).zfill(8),
                "conversations": [
                    {
                        "from": "user",
                        "value": prompt_template.replace("QUESTION", question),
                    },
                    {
                        "from": "assistant",
                        "value": f"城市:{city}\n日期:{time_str}",
                    },
                ],
            }
        )

    print("finish generate text samples.")

    print("start write generated samples to jsonl file.")
    json.dump(
        generated_samples,
        open("datas/generated_deepspeed_samples.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )

    print("finish write generated samples to jsonl file.")


if __name__ == "__main__":
    parser_parent = args_global()
    parser = argparse.ArgumentParser(parents=[parser_parent])
    parser.add_argument(
        "--train-finetune-execute-function",
        type=str,
        default="qwen_official_finutune_dataset_prepreare",
        help="train finetune 选择要执行的函数",
    )
    args = parser.parse_args()

    if (
        args.train_finetune_execute_function
        == "qwen_official_finutune_dataset_prepreare"
    ):
        qwen_official_finutune_dataset_prepreare(args=args)
    elif args.train_finetune_execute_function == "data_dealing":
        data_dealing()
