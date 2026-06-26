import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from math_utils import extract_boxed_answer, is_equivalent

BASE_MODEL_PATH = "models/qwen2.5-7b-pytorch"
NUM_PROBLEMS = 200
MAX_NEW_TOKENS = 512


def evaluate_model(model, tokenizer, dataset, num_samples):
    correct = 0
    total = 0
    results = []

    data = dataset['test'].select(range(num_samples))

    for i, example in enumerate(data):
        problem = example['problem']
        true_answer = example['answer']

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and put your final answer in \\boxed{}."},
            {"role": "user", "content": problem}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("mps")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted_answer = extract_boxed_answer(response)
        is_correct = is_equivalent(predicted_answer, true_answer)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            'problem': problem,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'correct': is_correct,
            'level': example['level'],
            'subject': example['subject'],
            'response': response,
        })

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_samples} | Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    accuracy = correct / total
    return accuracy, results


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

    print("Loading MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")

    print(f"\nEvaluating on {NUM_PROBLEMS} MATH-500 problems ({model_label})...\n")
    accuracy, results = evaluate_model(model, tokenizer, dataset, NUM_PROBLEMS)

    print(f"\n{'='*50}")
    print(f"RESULTS ({model_label})")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy*100:.1f}% ({sum(r['correct'] for r in results)}/{NUM_PROBLEMS})")

    # breakdown by difficulty level, since MATH levels range 1-5
    levels = sorted(set(r['level'] for r in results))
    for lvl in levels:
        lvl_results = [r for r in results if r['level'] == lvl]
        lvl_correct = sum(r['correct'] for r in lvl_results)
        print(f"  Level {lvl}: {lvl_correct}/{len(lvl_results)} ({100*lvl_correct/len(lvl_results):.1f}%)")

    output_path = f"results/math500_{model_label}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'model_label': model_label,
            'num_problems': NUM_PROBLEMS,
            'accuracy': accuracy,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
