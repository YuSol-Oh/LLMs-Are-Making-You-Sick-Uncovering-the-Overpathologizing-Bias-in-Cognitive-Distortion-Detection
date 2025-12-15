"""
finetune_distortion_classification.py

Supervised fine-tuning (SFT) of a Llama-3.1-8B-based classifier for
cognitive distortion type classification using QLoRA and TRL's SFTTrainer.

Inputs
------
- data/finetuning/finetuning_data_classification_train.json
- data/finetuning/finetuning_data_classification_valid.json

Each JSON file should be a list of objects with at least:
  {
    "text": "<instruction + answer style training example>"
  }

Outputs
-------
- Saved PEFT-adapted model under:
  models/distortion_classification/Llama-finetuned-Distortion-Classification-Model
- HF Trainer artifacts (logs, checkpoints) under:
  outputs/finetune_classification/
"""

import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# =========================
# Paths & constants
# =========================

DATA_DIR = os.path.join("data", "finetuning")
TRAIN_PATH = os.path.join(DATA_DIR, "finetuning_data_classification_train.json")
VALID_PATH = os.path.join(DATA_DIR, "finetuning_data_classification_valid.json")

OUTPUT_DIR = os.path.join("outputs", "finetune_classification")
MODEL_DIR = os.path.join(
    "models", "distortion_classification", "Llama-finetuned-Distortion-Classification-Model"
)

BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def main():
    # Optional: select a specific GPU (uncomment if you need it)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Basic GPU memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================
    # 0. Load datasets
    # =========================
    train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
    valid_dataset = load_dataset("json", data_files=VALID_PATH, split="train")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    # =========================
    # 1. QLoRA config
    # =========================
    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # =========================
    # 2. Load base model (LLaMA 3.1)
    # =========================
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=qlora_config,
        device_map={"": 0},  # or "auto" if you want automatic device mapping
    )

    # Recommended for training with gradient checkpointing / QLoRA
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # =========================
    # 3. Load tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        # If your base model requires auth, set HF_TOKEN in env and pass:
        # token=os.environ.get("HF_TOKEN", None),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Base model and tokenizer successfully loaded.")

    # =========================
    # 4. PEFT (LoRA) configuration
    # =========================
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # =========================
    # 5. TrainingArguments
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        # max_steps=...  # you can override epochs by using max_steps if needed
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,    # helps with memory usage
        optim="paged_adamw_8bit",
        learning_rate=1e-4,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        logging_steps=100,
    )

    # =========================
    # 6. Supervised fine-tuning with SFTTrainer
    # =========================
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # =========================
    # 7. Save fine-tuned model
    # =========================
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    trainer.save_model(MODEL_DIR)
    print(f"Fine-tuned model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
