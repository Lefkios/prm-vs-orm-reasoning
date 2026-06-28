import os
import json
import logging
import torch
import torch.nn.functional as F
import sys
from dotenv import load_dotenv
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

sys.path.insert(0, 'src')
from math_utils import extract_boxed_answer, is_equivalent

load_dotenv()

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)

CONFIG = {
    "model_path": "models/qwen2.5-7b-pytorch",
    "output_path": "models/prm-human-math-trained",
    "filtered_problems_path": "data/filtered_math_problems.json",
    "prm_classifier_path": "models/prm-classifier",
    "prm_classifier_base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "group_size": 8,
    "max_new_tokens": 384,
    "learning_rate": 1e-5,
    "num_epochs": 2,
    "clip_epsilon": 0.2,
    # raised from 0.01 — the GSM8K PRM-human run showed entropy collapse and
    # eventual numerical instability once the reward saturated; a stronger KL
    # penalty keeps the policy closer to the reference model and should slow
    # that collapse down
    "kl_coefficient": 0.05,
    "wandb_project": "prm-vs-orm-reasoning",
    "wandb_run_name": "prm-human-math-training",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

PRM_MODEL = None
PRM_TOKENIZER = None


def split_into_steps(text):
    lines = [line.strip() for line in text.split('\n')]
    return [line for line in lines if line]


def build_classifier_text(prompt, steps_so_far):
    joined = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps_so_far))
    return f"Problem: {prompt}\n\nSolution so far:\n{joined}"


def score_steps(prompt, steps):
    if not steps:
        return 0.0

    texts = [build_classifier_text(prompt, steps[:i + 1]) for i in range(len(steps))]
    inputs = PRM_TOKENIZER(
        texts, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to("mps")

    with torch.no_grad():
        logits = PRM_MODEL(**inputs).logits
        probs = F.softmax(logits, dim=-1)[:, 1]

    return probs.mean().item()


def prm_reward_function(prompts, completions, answer, **kwargs):
    rewards = []
    for prompt, completion, true_answer in zip(prompts, completions, answer):
        if isinstance(completion, list):
            text = completion[-1].get('content', '') if completion else ""
        elif isinstance(completion, dict):
            text = completion.get('content', '')
        else:
            text = str(completion)

        if isinstance(prompt, list):
            prompt_text = prompt[-1].get('content', '') if prompt else ""
        else:
            prompt_text = str(prompt)

        steps = split_into_steps(text)
        step_score = score_steps(prompt_text, steps)

        predicted = extract_boxed_answer(text)
        outcome_bonus = 0.5 if is_equivalent(predicted, true_answer) else -0.5

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
                {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step and put your final answer in \\boxed{}."},
                {"role": "user", "content": p["problem"]}
            ],
            "answer": p["answer"]
        })

    print(f"Loaded {len(formatted)} filtered MATH problems from {path}")
    return formatted


def main():
    global PRM_MODEL, PRM_TOKENIZER

    if not os.path.exists(CONFIG["filtered_problems_path"]):
        raise FileNotFoundError(
            f"{CONFIG['filtered_problems_path']} not found. Run src/filter_math_problems.py first."
        )
    if not os.path.exists(CONFIG["prm_classifier_path"]):
        raise FileNotFoundError(
            f"{CONFIG['prm_classifier_path']} not found. Run src/train_prm_classifier.py first."
        )

    train_data = load_filtered_dataset(CONFIG["filtered_problems_path"])

    print("\nLoading PRM classifier (reward model)...")
    PRM_TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["prm_classifier_path"])
    if PRM_TOKENIZER.pad_token is None:
        PRM_TOKENIZER.pad_token = PRM_TOKENIZER.eos_token

    prm_base = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["prm_classifier_base_model"],
        num_labels=2,
        dtype=torch.float16,
        device_map="mps",
    )
    prm_base.config.pad_token_id = PRM_TOKENIZER.pad_token_id
    PRM_MODEL = PeftModel.from_pretrained(prm_base, CONFIG["prm_classifier_path"])
    PRM_MODEL = PRM_MODEL.merge_and_unload()
    PRM_MODEL.config.pad_token_id = PRM_TOKENIZER.pad_token_id
    PRM_MODEL.eval()

    print("\nLoading policy model...")
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
        reward_funcs=prm_reward_function,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    print("\nStarting PRM-human-MATH training...")
    print(f"Problems: {len(train_data)}")
    print(f"Group size: {CONFIG['group_size']}")
    print(f"Monitor at: https://wandb.ai/lefkios-univeristy-of-cyprus/prm-vs-orm-reasoning\n")

    trainer.train()

    print("\nSaving trained model...")
    trainer.save_model(CONFIG["output_path"])
    tokenizer.save_pretrained(CONFIG["output_path"])
    print(f"Model saved to {CONFIG['output_path']}")
    print("PRM-human-MATH training complete.")


if __name__ == "__main__":
    main()
