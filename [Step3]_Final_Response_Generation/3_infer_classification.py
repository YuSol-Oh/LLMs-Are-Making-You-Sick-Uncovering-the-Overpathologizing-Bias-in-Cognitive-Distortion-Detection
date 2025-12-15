"""
inference_distortion_classification.py

Run inference using the fine-tuned distortion classification model.

Inputs
------
- data/inference/input_distortion_classification.json
    Training-style prompts for classification:
    [
      {
        "input_id": ...,
        "distorted_hop": ...,
        "input_prompt": "<instruction prompt>"
      },
      ...
    ]

Outputs
-------
- Saved inference results under:
  outputs/inference/distortion_classification_results.json
"""

import os
import json
import gc
import torch
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

# ========================================
# Paths
# ========================================

MODEL_DIR = os.path.join(
    "models", "distortion_classification", "Llama-finetuned-Distortion-Classification-Model"
)

INPUT_PATH = os.path.join(
    "data", "inference", "input_distortion_classification.json"
)

OUTPUT_PATH = os.path.join(
    "outputs", "inference", "distortion_classification_results.json"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ========================================
# GPU memory management
# ========================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
gc.collect()
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"


# ========================================
# Load model & tokenizer
# ========================================

print("Loading classification model...")

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model = LlamaForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

print("Model loaded.")


# ========================================
# Generation function
# ========================================

def generate_response(prompt: str) -> str:
    """Generate model output given a classification prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        inputs.input_ids,
        max_new_tokens=30,
        attention_mask=inputs.attention_mask,
        temperature=0.1,
    )

    decoded = tokenizer.batch_decode(output)[0]
    cleaned = decoded.replace("<|begin_of_text|>", "").replace(prompt, "")

    return cleaned.strip()


# ========================================
# Main inference loop
# ========================================

def main():
    print("Loading input data...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    print("Running inference...")
    for idx, item in enumerate(data):
        prompt = item["input_prompt"]
        inferenced_text = generate_response(prompt)

        results.append(
            {
                "input_id": item["input_id"],
                "distorted_hop": item.get("distorted_hop"),
                "inferenced": inferenced_text,
            }
        )

        print(f"[{idx+1}/{len(data)}] completed")

    print(f"Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
