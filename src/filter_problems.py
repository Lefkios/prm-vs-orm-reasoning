import re
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/qwen2.5-7b-pytorch"
TARGET_GOOD_PROBLEMS = 300
SAMPLES_PER_PROBLEM = 4
MAX_TOKENS_FILTER = 200
OUTPUT_PATH = "data/filtered_gsm8k_problems.json"


def extract_answer(text):
    match = re.search(r'####\s*([-\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    match = re.search(r'\\boxed\{([-\d,\.]+)\}', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    if numbers:
        return numbers[-1].strip()
    return None


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="mps")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset('openai/gsm8k', 'main')
    candidates = dataset['train'].shuffle(seed=42)

    good_problems = []
    skipped_all_correct = 0
    skipped_all_wrong = 0

    for i, example in enumerate(candidates):
        if len(good_problems) >= TARGET_GOOD_PROBLEMS:
            break

        question = example['question']
        true_answer = extract_answer(example['answer'])

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and end your answer with #### followed by the final number."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")

        results = []
        for _ in range(SAMPLES_PER_PROBLEM):
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS_FILTER,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
            text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_answer(text)
            results.append(predicted == true_answer)

        num_correct = sum(results)

        if num_correct == 0:
            skipped_all_wrong += 1
        elif num_correct == SAMPLES_PER_PROBLEM:
            skipped_all_correct += 1
        else:
            good_problems.append({
                "question": question,
                "answer": true_answer,
                "sample_correct_rate": num_correct / SAMPLES_PER_PROBLEM
            })

        if (i + 1) % 25 == 0:
            print(f"Checked {i+1} | "
                  f"Kept: {len(good_problems)}/{TARGET_GOOD_PROBLEMS} | "
                  f"All-correct skipped: {skipped_all_correct} | "
                  f"All-wrong skipped: {skipped_all_wrong}")
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(good_problems, f, indent=2)

    print(f"\nDone. Found {len(good_problems)} problems with mixed correctness.")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(good_problems, f, indent=2)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()