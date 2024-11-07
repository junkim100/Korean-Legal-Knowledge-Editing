import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict
from tqdm import tqdm
import fire


def extract_raw_data(raw_data_dir: str, verbose: bool) -> dict:
    """Extracts label and full_text pairs from JSON files in the specified directory."""
    result = {}
    total_count = 0
    for file in os.listdir(raw_data_dir):
        if file.endswith(".json"):
            with open(os.path.join(raw_data_dir, file), "r") as f:
                data = json.load(f)
                count = 0
                for i, key in enumerate(data):
                    try:
                        label = data[key]["label"]
                        full_text = data[key]["fullText"]
                        if len(full_text) < 2 * len(label):
                            continue
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
    total_count += count
    return result, total_count


def create_messages(label: str, full_text: str) -> dict:
    """Generates a response based on the provided label using a pre-trained model."""
    messages_dict = {}
    messages_1 = [
        {"role": "system", "content": "다음 법령 조항이 무엇인지 알려주세요."},
        {"role": "user", "content": "119긴급신고법 제18조의 제1항"},
        {
            "role": "assistant",
            "content": "① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다. ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다.",
        },
        {"role": "user", "content": label},
    ]

    messages_2 = [
        {
            "role": "system",
            "content": "다음 법령 조항 설명을 읽고 법률의 이름을 알려주세요.",
        },
        {
            "role": "user",
            "content": "① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다. ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다.",
        },
        {"role": "assistant", "content": "119긴급신고법 제18조의 제1항"},
        {"role": "user", "content": full_text},
    ]

    messages_dict["type1"] = messages_1
    messages_dict["type2"] = messages_2

    return messages_dict


def calculate_bleu(reference, hypothesis):
    chencherry = SmoothingFunction()
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Define length thresholds
    SHORT_TEXT = 10  # tokens
    MEDIUM_TEXT = 30  # tokens

    # Get text lengths
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    max_len = max(ref_len, hyp_len)

    # Choose n-gram weights based on text length
    if max_len < SHORT_TEXT:
        # For very short texts, focus on unigrams and bigrams
        weights = (0.6, 0.4, 0, 0)
    elif max_len < MEDIUM_TEXT:
        # For medium texts, include trigrams
        weights = (0.4, 0.3, 0.3, 0)
    else:
        # For longer texts, use all n-grams
        weights = (0.25, 0.25, 0.25, 0.25)

    return sentence_bleu(
        [ref_tokens], hyp_tokens, weights=weights, smoothing_function=chencherry.method1
    )


def prompt_variation(
    model_id: str,
    save_dir: str,
    raw_data_dir: str,
    verbose: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Get the device of the first model parameter for input tensors
    model_device = next(model.parameters()).device

    # TODO: Make this more robust
    # Clear the results directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))

    type1_entries = []
    type2_entries = []

    id = 0
    raw_data, total_count = extract_raw_data(raw_data_dir, verbose)
    bar = tqdm(total=total_count, desc="Processing entries", unit="entry")
    for label, full_text in raw_data.items():
        messages_dict = create_messages(label, full_text)
        for key, messages in messages_dict.items():
            print(key, messages)
            # Encode input with tokenizer
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
            )

            # Create attention mask
            attention_mask = torch.ones_like(encoded)

            # Move tensors to same device as model
            inputs = encoded.to(model_device)
            attention_mask = attention_mask.to(model_device)

            # Generate response
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Only keep the assistant's response
            response = response.split("assistant\n\n")[-1]

            # Calculate BLEU score
            bleu_score = calculate_bleu(full_text, response)

            # Create entry dictionary
            if key == "type1":
                entry = {
                    "id": id,
                    "prompt": label,
                    "response": response,
                    "golden": full_text,
                    "bleu": bleu_score,
                }
                type1_entries.append(entry)
            elif key == "type2":
                entry = {
                    "id": id,
                    "prompt": full_text,
                    "response": response,
                    "golden": label,
                    "bleu": bleu_score,
                }
                type2_entries.append(entry)
            else:
                print(f"Invalid key: {key}")
                continue

            if verbose:
                print(f"Processed entry for label: {label}")

        id += 1
        bar.update(1)

    # Save accumulated results to JSON files
    try:
        if type1_entries:
            with open(f"{save_dir}/type1.json", "w", encoding="utf-8") as f:
                json.dump(type1_entries, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"Results saved to {save_dir}/type1.json")

        if type2_entries:
            with open(f"{save_dir}/type2.json", "w", encoding="utf-8") as f:
                json.dump(type2_entries, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"Results saved to {save_dir}/type2.json")
    except Exception as e:
        print(f"Error saving results: {e}")


def sort_avg(save_dir: str) -> List[Dict]:
    bleu_data = {}
    entry_data = {}  # Store complete entry information

    # Read all JSON files in the save directory
    for filename in os.listdir(save_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(save_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
                for entry in entries:
                    entry_id = entry["id"]
                    bleu_score = entry["bleu"]

                    # Store BLEU scores for averaging
                    if entry_id not in bleu_data:
                        bleu_data[entry_id] = []
                        entry_data[entry_id] = entry  # Store complete entry
                    bleu_data[entry_id].append(bleu_score)

    avg_bleu_data = []

    # Calculate average BLEU scores and create new entries
    for entry_id, scores in bleu_data.items():
        avg_bleu = sum(scores) / len(scores)
        # Create new entry with original data plus average BLEU
        new_entry = {
            "id": entry_id,
            "avg_bleu": avg_bleu,
        }
        avg_bleu_data.append(new_entry)

    # Sort entries by average BLEU score in descending order
    sorted_avg_bleu_data = sorted(
        avg_bleu_data, key=lambda x: x["avg_bleu"], reverse=True
    )

    # Save sorted data to JSON file
    output_file = os.path.join(save_dir, "sorted_avg.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_avg_bleu_data, f, ensure_ascii=False, indent=2)

    return sorted_avg_bleu_data


def main(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    save_dir: str = "./results",
    raw_data_dir: str = "./법령지식",
    verbose: bool = False,
):
    prompt_variation(model_id, save_dir, raw_data_dir, verbose)
    sort_avg(save_dir)


if __name__ == "__main__":
    fire.Fire(main)
