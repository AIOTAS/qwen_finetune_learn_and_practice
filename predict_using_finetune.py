from transformers import AutoTokenizer, AutoModelForCausalLM
from args_g import args_global
import torch


def predict_using_finetune():
    parser = args_global()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.finetuned_predict_model_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_predict_model_path, device_map="auto", trust_remote_code=True
    ).eval()

    model.generation_config.top_p = 0  # 只选择概率最高的token

    with open(args.finetune_prompt_template_filepath, "r", encoding="utf_8_sig") as fr:
        prompt_template = fr.read()

    question_list = [
        "2024年10月2号在安徽阜阳太和县的天气很晴朗",
        "2025年1月18日在北京的天气很好",
        "河北1月26号肯定天气很棒",
    ]

    for question in question_list:
        prompt = prompt_template.replace("QUESTION", question)

        with torch.no_grad():
            response, history = model.chat(tokenizer, prompt, history=None)
            print(response)

    response, history = model.chat(tokenizer, "你是谁", history=None)
    print(response)


if __name__ == "__main__":
    predict_using_finetune()
