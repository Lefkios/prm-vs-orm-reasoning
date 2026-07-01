import re
import os
import json
import logging
import torch
import sys
from dotenv import load_dotenv
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, 'src')
from reward_functions.auto_labels import score_completions_batch

load_dotenv()

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY not set — add it to .env before running.")

CONFIG = {
    "model_path": "models/qwen2.5-7b-pytorch",
    "output_path": "models/prm-auto-gsm8k-trained",
    "filtered_problems_path": "data/filtered_gsm8k_problems.json",
    "judge_model": "gpt-4o-mini",
    "group_size": 8,
    "max_new_tokens": 384,
    "learning_rate": 1e-5,
    "num_epochs": 2,
    "clip_epsilon": 0.2,
    "kl_coefficient": 0.05,
    "wandb_project": "prm-vs-orm-reasoning",
    "wandb_run_name": "prm-auto-gsm8k-training",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}


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


def split_into_steps(text):
    lines = [line.strip() for line in text.split('\n')]
    return [line for line in lines if line]


def prm_auto_reward_function(prompts, completions, answer, **kwargs):
    texts = []
    for completion in completions:
        if isinstance(completion, list):
            texts.append(completion[-1].get('content', '') if completion else "")
        elif isinstance(completion, dict):
            texts.append(completion.get('content', ''))
        else:
            texts.append(str(completion))

    prompt_texts = []
    for prompt in prompts:
        if isinstance(prompt, list):
            prompt_texts.append(prompt[-1].get('content', '') if prompt else "")
        else:
            prompt_texts.append(str(prompt))

    steps_list = [split_into_steps(t) for t in texts]
    step_scores = score_completions_batch(
        list(zip(prompt_texts, steps_list)), model=CONFIG["judge_model"]
    )

    rewards = []
    for text, true_answer, step_score in zip(texts, answer, step_scores):
        predicted = extract_answer(text)
        outcome_bonus = 0.5 if predicted == true_answer else -0.5
        reward = (2 * step_score - 1) + outcome_bonus
        rewards.append(reward)
    return rewards


def load_filtered_dataset(path):
    with open(path) as f:
        problems = json.load(f)

    formatted = []
    for p in problems:
        formatted.append({
            "prompt": [
                {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and end your answer with #### followed by the final number."},
                {"role": "user", "content": p["question"]}
            ],
            "answer": p["answer"]
        })

    print(f"Loaded {len(formatted)} filtered GSM8K problems from {path}")
    return formatted


def main():
    if not os.path.exists(CONFIG["filtered_problems_path"]):
        raise FileNotFoundError(
            f"{CONFIG['filtered_problems_path']} not found. Run src/filter_problems.py first."
        )

    train_data = load_filtered_dataset(CONFIG["filtered_problems_path"])

    print("\nLoading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"], dtype=torch.float16, device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    grpo_config = GRPOConfig(
        output_dir=CONFIG["output_path"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=1,
        num_generations=CONFIG["group_size"],
        generation_batch_size=8,
        max_completion_length=CONFIG["max_new_tokens"],
        learning_rate=CONFIG["learning_rate"],
        beta=CONFIG["kl_coefficient"],
        epsilon=CONFIG["clip_epsilon"],
        logging_steps=10,
        save_steps=100,
        report_to="wandb",
        run_name=CONFIG["wandb_run_name"],
        temperature=1.0,
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model, args=grpo_config,
        reward_funcs=prm_auto_reward_function,
        train_dataset=train_data, processing_class=tokenizer,
    )

    print("\nStarting PRM-auto-GSM8K training...")
    print(f"Problems: {len(train_data)}")
    print(f"Group size: {CONFIG['group_size']}")
    print(f"Judge model: {CONFIG['judge_model']}")
    print(f"Monitor at: https://wandb.ai/lefkios-univeristy-of-cyprus/prm-vs-orm-reasoning\n")

    trainer.train()

    print("\nSaving trained model...")
    trainer.save_model(CONFIG["output_path"])
    tokenizer.save_pretrained(CONFIG["output_path"])
    print(f"Model saved to {CONFIG['output_path']}")
    print("PRM-auto-GSM8K training complete.")


if __name__ == "__main__":
    main()
