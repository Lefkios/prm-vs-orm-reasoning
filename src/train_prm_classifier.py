"""
Train a step-level correctness classifier on PRM800K labels.

Uses Qwen2.5-0.5B with a classification head (AutoModelForSequenceClassification).
Each training example is (problem + partial solution up to step N) -> label {0, 1}.

Dataset format (HuggingFaceH4/prm800k-trl-dedup):
  prompt      : str   — the math problem
  completions : list  — list of step strings
  labels      : list  — list of ints (1=correct, -1=wrong, 0=neutral/skip)
  index       : int

We flatten each (problem, prefix, label) triple into an independent row,
keeping only steps with label 1 or -1 (skip neutral 0s).
"""

import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "output_path": "models/prm-classifier",
    "max_length": 512,
    "batch_size": 8,
    "num_epochs": 2,
    "learning_rate": 2e-5,
    "max_train_examples": 50_000,  # enough to get a solid classifier; full set is 369k
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}


def build_text(prompt, steps_so_far):
    """Concatenate problem + steps seen so far into a single string for classification."""
    joined = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps_so_far))
    return f"Problem: {prompt}\n\nSolution so far:\n{joined}"


def flatten_prm800k(split_data, max_examples):
    """Turn each (problem, completions, labels) row into step-level examples.
    Labels in this dataset are plain booleans (True=correct, False=incorrect),
    not the 1/-1/0 scheme the README implies — there's no neutral label here."""
    rows = []
    for item in split_data:
        prompt = item["prompt"]
        completions = item["completions"]
        labels = item["labels"]

        steps_so_far = []
        for step, label in zip(completions, labels):
            steps_so_far.append(step)
            rows.append({"text": build_text(prompt, steps_so_far), "label": int(bool(label))})

            if len(rows) >= max_examples:
                return rows
    return rows


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    # also report class-wise accuracy since PRM800K is ~85% positive
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acc = (preds[pos_mask] == 1).mean() if pos_mask.any() else 0.0
    neg_acc = (preds[neg_mask] == 0).mean() if neg_mask.any() else 0.0
    return {"accuracy": acc, "pos_acc": pos_acc, "neg_acc": neg_acc}


def main():
    print("Loading PRM800K...")
    raw = load_dataset("HuggingFaceH4/prm800k-trl-dedup")
    train_rows = flatten_prm800k(raw["train"], CONFIG["max_train_examples"])
    test_rows = flatten_prm800k(raw["test"], max_examples=2000)

    print(f"  Train examples: {len(train_rows)}")
    print(f"  Test examples:  {len(test_rows)}")
    pos_frac = sum(r["label"] for r in train_rows) / len(train_rows)
    print(f"  Positive label fraction: {pos_frac:.2%}")

    train_ds = Dataset.from_list(train_rows)
    test_ds = Dataset.from_list(test_rows)

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # num_labels=2: 0=wrong step, 1=correct step
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=2,
        dtype=torch.float32,
        device_map="mps",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=False,  # DataCollatorWithPadding handles dynamic padding
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=CONFIG["output_path"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nTraining PRM classifier...")
    trainer.train()

    print("\nSaving classifier...")
    trainer.save_model(CONFIG["output_path"])
    tokenizer.save_pretrained(CONFIG["output_path"])
    print(f"Saved to {CONFIG['output_path']}")


if __name__ == "__main__":
    main()
