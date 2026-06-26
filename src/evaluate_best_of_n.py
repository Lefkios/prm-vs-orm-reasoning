"""
Best-of-N evaluation: generate N samples per problem and measure two things:
  - majority-vote accuracy (self-consistency: most common answer wins)
  - pass@N (oracle upper bound: counts as correct if ANY of the N samples is correct)

Usage:
  python src/evaluate_best_of_n.py                          # evaluates base model
  python src/evaluate_best_of_n.py models/orm-trained/checkpoint-2400   # evaluates ORM checkpoint
"""

import re
import sys
import json
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "models/qwen2.5-7b-pytorch"
NUM_PROBLEMS = 200
N_SAMPLES = 8
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.8


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


def majority_vote(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def evaluate_best_of_n(model, tokenizer, dataset, num_problems, n_samples, split='test'):
    majority_correct = 0
    oracle_correct = 0
    total = 0
    results = []

    data = dataset[split].select(range(num_problems))

    for i, example in enumerate(data):
        question = example['question']
        true_answer = extract_answer(example['answer'])

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and end your answer with #### followed by the final number."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")

        sample_answers = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            sample_answers.append(extract_answer(response))

        voted = majority_vote(sample_answers)
        is_majority_correct = (voted == true_answer)
        is_oracle_correct = any(a == true_answer for a in sample_answers)

        if is_majority_correct:
            majority_correct += 1
        if is_oracle_correct:
            oracle_correct += 1
        total += 1

        results.append({
            'question': question,
            'true_answer': true_answer,
            'sample_answers': sample_answers,
            'majority_vote': voted,
            'majority_correct': is_majority_correct,
            'oracle_correct': is_oracle_correct,
        })

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_problems} | "
                  f"Majority-vote acc: {100*majority_correct/total:.1f}% | "
                  f"Pass@{n_samples} (oracle): {100*oracle_correct/total:.1f}%")

    return majority_correct / total, oracle_correct / total, results


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_label = "baseline"
    if checkpoint_path:
        print(f"Applying LoRA adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        model_label = checkpoint_path.replace("/", "_")

    print("Loading dataset...")
    dataset = load_dataset('openai/gsm8k', 'main')

    print(f"\nEvaluating best-of-{N_SAMPLES} on {NUM_PROBLEMS} GSM8K test problems ({model_label})...\n")
    majority_acc, oracle_acc, results = evaluate_best_of_n(
        model, tokenizer, dataset, NUM_PROBLEMS, N_SAMPLES
    )

    print(f"\n{'='*50}")
    print(f"RESULTS ({model_label})")
    print(f"{'='*50}")
    print(f"Majority-vote accuracy: {majority_acc*100:.1f}%")
    print(f"Pass@{N_SAMPLES} (oracle):     {oracle_acc*100:.1f}%")

    output_path = f"results/best_of_n_{model_label}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'model_label': model_label,
            'num_problems': NUM_PROBLEMS,
            'n_samples': N_SAMPLES,
            'majority_vote_accuracy': majority_acc,
            'pass_at_n_accuracy': oracle_acc,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
