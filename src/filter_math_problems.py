import json
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, 'src')
from math_utils import extract_boxed_answer, is_equivalent

MODEL_PATH = "models/qwen2.5-7b-pytorch"
TARGET_GOOD_PROBLEMS = 350
SAMPLES_PER_PROBLEM = 4
MAX_TOKENS_FILTER = 300
OUTPUT_PATH = "data/filtered_math_problems.json"
MATH_SUBJECTS = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus',
]


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="mps")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading MATH train set...")
    parts = [load_dataset('EleutherAI/hendrycks_math', s)['train'] for s in MATH_SUBJECTS]
    dataset = concatenate_datasets(parts)
    # exclude level 1 only: at ~95% baseline accuracy, nearly every group of 4
    # samples is all-correct, producing almost no mixed-correctness signal.
    # level 5, despite being hardest (~8% accuracy), still yields mixed groups
    # often enough (~28% of the time, by binomial chance) to be worth keeping.
    dataset = dataset.filter(lambda ex: ex['level'] != 'Level 1')
    dataset = dataset.shuffle(seed=42)

    good_problems = []
    skipped_all_correct = 0
    skipped_all_wrong = 0
    skipped_no_answer = 0

    for i, example in enumerate(dataset):
        if len(good_problems) >= TARGET_GOOD_PROBLEMS:
            break

        problem = example['problem']
        true_answer = extract_boxed_answer(example['solution'])
        if true_answer is None:
            skipped_no_answer += 1
            continue

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and put your final answer in \\boxed{}."},
            {"role": "user", "content": problem}
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
            predicted = extract_boxed_answer(text)
            results.append(is_equivalent(predicted, true_answer))

        num_correct = sum(results)

        if num_correct == 0:
            skipped_all_wrong += 1
        elif num_correct == SAMPLES_PER_PROBLEM:
            skipped_all_correct += 1
        else:
            good_problems.append({
                "problem": problem,
                "answer": true_answer,
                "level": example['level'],
                "sample_correct_rate": num_correct / SAMPLES_PER_PROBLEM
            })

        if (i + 1) % 25 == 0:
            print(f"Checked {i+1} | "
                  f"Kept: {len(good_problems)}/{TARGET_GOOD_PROBLEMS} | "
                  f"All-correct: {skipped_all_correct} | "
                  f"All-wrong: {skipped_all_wrong} | "
                  f"No-answer: {skipped_no_answer}")
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(good_problems, f, indent=2)

    print(f"\nDone. Found {len(good_problems)} problems with mixed correctness.")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(good_problems, f, indent=2)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
