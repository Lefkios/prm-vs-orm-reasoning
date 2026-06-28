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
from math_utils import extract_boxed_answer, is_equivalent

load_dotenv()

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)

CONFIG = {
    "model_path": "models/qwen2.5-7b-pytorch",
    "output_path": "models/orm-math-trained",
    "filtered_problems_path": "data/filtered_math_problems.json",
    "group_size": 8,
    "max_new_tokens": 384,
    "learning_rate": 1e-5,
    "num_epochs": 3,
    "clip_epsilon": 0.2,
    "kl_coefficient": 0.01,
    "wandb_project": "prm-vs-orm-reasoning",
    "wandb_run_name": "orm-math-training",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}


def orm_reward_function(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, true_answer in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[-1].get('content', '') if completion else ""
        elif isinstance(completion, dict):
            text = completion.get('content', '')
        else:
            text = str(completion)
        predicted = extract_boxed_answer(text)
        reward = 1.0 if is_equivalent(predicted, true_answer) else -1.0
        rewards.append(reward)
    return rewards


def load_filtered_dataset(path):
    with open(path) as f:
        problems = json.load(f)

    formatted = []
    for p in problems:
        formatted.append({
            "prompt": [
                {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and put your final answer in \\boxed{}."},
                {"role": "user", "content": p["problem"]}
            ],
            "answer": p["answer"]
        })

    print(f"Loaded {len(formatted)} filtered MATH problems from {path}")
    rates = [p["sample_correct_rate"] for p in problems]
    buckets = {0.25: 0, 0.5: 0, 0.75: 0}
    for r in rates:
        if r in buckets:
            buckets[r] += 1
    print(f"  Correct-rate distribution: {buckets}")

    levels = {}
    for p in problems:
        levels[p["level"]] = levels.get(p["level"], 0) + 1
    print(f"  Level distribution: {levels}")
    return formatted


def main():
    if not os.path.exists(CONFIG["filtered_problems_path"]):
        raise FileNotFoundError(
            f"{CONFIG['filtered_problems_path']} not found. "
            "Run src/filter_math_problems.py first."
        )

    train_data = load_filtered_dataset(CONFIG["filtered_problems_path"])

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        dtype=torch.float16,
        device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Configuring GRPO trainer...")
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
        model=model,
        args=grpo_config,
        reward_funcs=orm_reward_function,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    print("\nStarting ORM-MATH training...")
    print(f"Problems: {len(train_data)}")
    print(f"Group size: {CONFIG['group_size']}")
    print(f"Monitor at: https://wandb.ai/lefkios-univeristy-of-cyprus/prm-vs-orm-reasoning\n")

    trainer.train()

    print("\nSaving trained model...")
    trainer.save_model(CONFIG["output_path"])
    tokenizer.save_pretrained(CONFIG["output_path"])
    print(f"Model saved to {CONFIG['output_path']}")
    print("ORM-MATH training complete.")


if __name__ == "__main__":
    main()
