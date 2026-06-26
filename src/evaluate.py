import re
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/qwen2.5-7b-pytorch"

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

def evaluate_model(model, tokenizer, dataset, num_samples=100, split='test'):
    correct = 0
    total = 0
    results = []

    data = dataset[split].select(range(num_samples))

    for i, example in enumerate(data):
        question = example['question']
        true_answer = extract_answer(example['answer'])

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and end your answer with #### followed by the final number."},
            {"role": "user", "content": question}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to("mps")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        predicted_answer = extract_answer(response)
        is_correct = (predicted_answer == true_answer)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            'question': question,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'correct': is_correct,
            'response': response
        })

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_samples} | Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    accuracy = correct / total
    return accuracy, results


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset('openai/gsm8k', 'main')

    print("\nEvaluating on 100 GSM8K test problems...\n")

    accuracy, results = evaluate_model(
        model, tokenizer, dataset,
        num_samples=100,
        split='test'
    )

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy*100:.1f}% ({sum(r['correct'] for r in results)}/100)")

    with open('results/evaluation_results.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'num_samples': 100,
            'model': MODEL_PATH,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to results/evaluation_results.json")

    correct_examples = [r for r in results if r['correct']]
    if correct_examples:
        ex = correct_examples[0]
        print(f"\n--- Sample correct ---")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Predicted: {ex['predicted_answer']} | True: {ex['true_answer']} ✓")

    wrong_examples = [r for r in results if not r['correct']]
    if wrong_examples:
        ex = wrong_examples[0]
        print(f"\n--- Sample incorrect ---")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Predicted: {ex['predicted_answer']} | True: {ex['true_answer']} ✗")


if __name__ == "__main__":
    main()