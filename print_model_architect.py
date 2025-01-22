from transformers import AutoTokenizer, AutoModelForCausalLM
from args_g import args_global


def print_model_architect():
    parser = args_global()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name, device_map="auto", trust_remote_code=True
    )
    print(model)


if __name__ == "__main__":
    print_model_architect()
