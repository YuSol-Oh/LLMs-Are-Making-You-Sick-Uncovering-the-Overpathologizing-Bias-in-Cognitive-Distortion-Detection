"""
finetune_overpathologize_detector.py

Supervised fine-tuning (SFT) of the Overpathologize Detection Model (OD)
using QLoRA on Meta-Llama-3.1-8B-Instruct with TRL's SFTTrainer.

The model is trained to answer whether a given candidate causal thought is
reasonably included among five plausible causal thoughts for a target thought.
(The dataset text already contains the full instruction + answer.)

Inputs
------
- data/finetune_OD/finetuning_data_overpathologize_detection_model_ver02_train.json
- data/finetune_OD/finetuning_data_overpathologize_detection_model_ver02_valid.json

Each file is a JSON list with items of the form:
    {"text": "[INST] ... [/INST] Answer: overpathologized"}
or
    {"text": "[INST] ... [/INST] Answer: not overpathologized"}

Outputs
-------
- outputs/finetune_OD/ : Hugging Face Trainer artifacts (checkpoints, logs)
- LORA adapter weights saved under `NEW_MODEL_NAME` (by default:
  "Llama-finetuned-Overpathologize_Detection_Model")

Notes
-----
- This script uses:
    - QLoRA (4-bit quantization with BitsAndBytesConfig)
    - LoRA (PEFT) for parameter-efficient fine-tuning
    - TRL's SFTTrainer for instruction-style fine-tuning
- GPU selection and Hugging Face authentication are expected to be configured
  via environment variables:
    - CUDA_VISIBLE_DEVICES (optional)
    - HF_TOKEN (optional, if you need gated model access or pushing to hub)
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
# from huggingface_hub import login  # Uncomment if you need to login explicitly


# =========================
# Paths & configuration
# =========================

DATA_DIR = "data"
FINETUNE_DIR = os.path.join(DATA_DIR, "finetune_OD")
OUTPUT_DIR = os.path.join("outputs", "finetune_OD")

os.makedirs(FINETUNE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(
    FINETUNE_DIR,
    "finetuning_data_overpathologize_detection_model_ver02_train.json",
)
VALID_PATH = os.path.join(
    FINETUNE_DIR,
    "finetuning_data_overpathologize_detection_model_ver02_valid.json",
)

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NEW_MODEL_NAME = "Llama-finetuned-Overpathologize_Detection_Model"

# Optional: set these externally (recommended)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Optional: Hugging Face login (use environment variable HF_TOKEN)
# hf_token = os.environ.get("HF_TOKEN")
# if hf_token:
#     login(token=hf_token)


# =========================
# Helper functions
# =========================

def cleanup_cuda() -> None:
    """Clear CUDA cache and run Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_datasets(train_path: str, valid_path: str):
    """Load train and validation datasets from JSON files."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Validation file not found: {valid_path}")

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    valid_dataset = load_dataset("json", data_files=valid_path, split="train")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    return train_dataset, valid_dataset


def create_qlora_config() -> BitsAndBytesConfig:
    """Create a 4-bit QLoRA quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )


def load_model_and_tokenizer(qlora_config: BitsAndBytesConfig):
    """Load base LLaMA model and tokenizer with QLoRA config."""
    print("Loading base model and tokenizer...")

    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=qlora_config,
        device_map="auto",  # Let HF/Accelerate infer device placement
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        # token=os.environ.get("HF_TOKEN"),  # Optional: if model access is gated
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Model and tokenizer loaded.")
    return model, tokenizer


def create_lora_config() -> LoraConfig:
    """Create LoRA PEFT configuration."""
    return LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_training_arguments() -> TrainingArguments:
    """Create TrainingArguments for SFTTrainer."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,  # distribute eval memory
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
        report_to="tensorboard",
        logging_steps=50,
    )


def train_overpathologize_detector():
    """Main training procedure for the Overpathologize Detection Model."""
    cleanup_cuda()

    # 0. Load datasets
    train_dataset, valid_dataset = load_datasets(TRAIN_PATH, VALID_PATH)

    # 1. QLoRA configuration
    qlora_config = create_qlora_config()

    # 2. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(qlora_config)

    # 3. PEFT / LoRA configuration
    peft_params = create_lora_config()

    # 4. Training arguments
    training_params = create_training_arguments()

    # 5. SFTTrainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 6. Save fine-tuned model (LoRA adapter + config)
    trainer.save_model(NEW_MODEL_NAME)
    print(f"Fine-tuned model saved to: {NEW_MODEL_NAME}")


if __name__ == "__main__":
    train_overpathologize_detector()
