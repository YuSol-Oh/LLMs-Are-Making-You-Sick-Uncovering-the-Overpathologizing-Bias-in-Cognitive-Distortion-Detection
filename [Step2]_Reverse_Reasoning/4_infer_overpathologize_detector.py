"""
infer_overpathologize_detector.py

Run inference with the fine-tuned Overpathologize Detection Model (OD).

Inputs
------
- MODEL_DIR:
    Directory containing the fine-tuned model:
        "Llama-finetuned-Overpathologize_Detection_Model"
    (saved by the SFT training script)

- ODM input file:
    data/reverse_reasoning/input_ODM_gpt-4omini.json

    Each item should have the following structure:
        {
            "input_id": ...,
            "input_causal_thought": "...",
            "input_result_thought": "...",
            "input_prompt": "You will be given the following: ..."
        }

Outputs
-------
- outputs/reverse_reasoning/odm_predictions.json

    Each item:
        {
            "input_id": ...,
            "input_causal_thought": "...",
            "input_result_thought": "...",
            "inferenced": "overpathologized" / "not overpathologized" / raw text
        }
"""

import os
import gc
import json
from typing import List, Dict

import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
)


# =========================
# Paths & configuration
# =========================

MODEL_DIR = "Llama-finetuned-Overpathologize_Detection_Model"

DATA_DIR = "data"
STEP2_DIR = os.path.join(DATA_DIR, "reverse_reasoning")
INPUT_ODM_PATH = os.path.join(STEP2_DIR, "input_ODM_gpt-4omini.json")

OUTPUT_DIR = os.path.join("outputs", "reverse_reasoning")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_ODM_PREDICTIONS = os.path.join(OUTPUT_DIR, "odm_predictions.json")

# Optional: configure CUDA memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

# Optional: Hugging Face login if needed (gated model / hub push)
# from huggingface_hub import login
# hf_token = os.environ.get("HF_TOKEN")
# if hf_token:
#     login(token=hf_token)


# =========================
# Utilities
# =========================

def cleanup_cuda() -> None:
    """Clear CUDA cache and run Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_odm_input(path: str) -> List[Dict]:
    """Load ODM inference input file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ODM input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} ODM input items.")
    return data


def load_model_and_tokenizer():
    """Load fine-tuned OD model and tokenizer."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Make sure you ran the SFT training script and saved the model there."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    model = LlamaForCausalLM.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, device


def generate_response(
    model: LlamaForCausalLM,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 0.3,
) -> str:
    """Generate a short answer from the OD model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

    # Remove possible BOS token and original prompt from the decoded text
    decoded = decoded.replace("<|begin_of_text|>", "")
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]

    # Strip special tokens and whitespace
    decoded = decoded.replace(tokenizer.eos_token or "", "").strip()
    return decoded


# =========================
# Main inference loop
# =========================

def run_inference():
    cleanup_cuda()

    # 1. Load data & model
    odm_inputs = load_odm_input(INPUT_ODM_PATH)
    model, tokenizer, device = load_model_and_tokenizer()

    results = []

    for idx, item in enumerate(odm_inputs):
        instruction = item["input_prompt"]
        # Encourage the model not to explain the reasoning
        prompt = f"{instruction} (Don't answer the reason)"

        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.3,
        )

        result_item = {
            "input_id": item["input_id"],
            "input_causal_thought": item["input_causal_thought"],
            "input_result_thought": item["input_result_thought"],
            "inferenced": answer,
        }
        results.append(result_item)

        print(f"[{idx + 1}/{len(odm_inputs)}] Done (id={item['input_id']})")

    # 2. Save predictions
    with open(OUTPUT_ODM_PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Inference completed. Saved predictions to: {OUTPUT_ODM_PREDICTIONS}")


if __name__ == "__main__":
    run_inference()
