import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json


def extract_raw_data(
    raw_data_dir: str = "/home/work/jun/Korean-Legal-Knowledge-Editing/법령지식",
    verbose: bool = False,
) -> dict:
    result = {}  # dictionary to store label and full_text pairs

    for file in os.listdir(raw_data_dir):
        if file.endswith(".json"):
            with open(
                f"/home/work/jun/Korean-Legal-Knowledge-Editing/법령지식/{file}", "r"
            ) as f:
                data = json.load(f)

                count = 0

                for i in range(len(data)):
                    try:
                        # Get the first (and only) key from the JSON
                        main_key = list(data.keys())[i]

                        # Extract the label
                        label = data[main_key][
                            "http://www.w3.org/2000/01/rdf-schema#label"
                        ][0]["value"]

                        # Extract the full text
                        full_text = data[main_key][
                            "http://www.aihub.or.kr/kb/law/fullText"
                        ][0]["value"]

                        # Save the pair into the result dictionary
                        result[label] = full_text

                        count += 1
                    except Exception as e:
                        if verbose:
                            print(f"Exception at {i} in {file}: {e}")
                        continue

                if verbose:
                    print(
                        f"{count} out of {len(data)} were successfully extracted from {file}"
                    )

    return result


def create_prompt(
    label: str, full_text: str, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
) -> str:
    messages = [
        {"role": "system", "content": "다음 법령 조항이 무엇인지 알려주세요."},
        {"role": "user", "content": "자동차손해배상 보장법 제45조의2 제1항"},
        {
            "role": "assistant",
            "content": "제45조의2 (정보의 제공 및 관리)  ① 제45조제3항에 따라 업무를 위탁받은 보험요율산출기관은 같은 조 제1항에 따라 업무를 위탁받은 자의 요청이 있는 경우 제공할 정보의 내용 등 대통령령으로 정하는 범위에서 가입관리전산망에서 관리되는 정보를 제공할 수 있다.",
        },
        {"role": "user", "content": label},
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to("cuda")

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors=True)
    outputs = model.generate(
        inputs, max_new_tokens=256, temperature=0.7, do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    entries = []
    for label, full_text in extract_raw_data().items():
        entry = {
            "label": label,
            "response": create_prompt(label, full_text),
            "golden truth": full_text,
        }
        entries.append(entry)

    with open("entries.json", "w") as f:
        json.dump(entries, f)
