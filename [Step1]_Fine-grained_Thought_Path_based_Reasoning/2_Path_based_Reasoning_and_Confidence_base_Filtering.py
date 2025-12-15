"""
2_Path_Based_Reasoning_and_Confidence_Filtering.py

Step 1 in REFORM: Path-based Reasoning + Confidence-based Filtering.

This script:
    1) Uses fine-grained Thought Paths and Cognitive Models
       to perform path-based distortion assessment with an LLM.
    2) Extracts log-probabilities to build soft scores over "yes"/"no"
       and marks which cases need Reverse Reasoning (need_check="yes").
    3) Maps the predicted distorted hop back to the underlying
       causal/result thoughts.
    4) Evaluates detection performance (yes/no) as a first-stage metric.

Inputs
------
- data/Cognitive_Distortion_Detection.json
    Original dataset with fields such as:
    - Id_Number
    - Patient Question
    - Distorted part
    - Dominant Distortion
    - Secondary Distortion (Optional)

- outputs/refom_inference/(step1)_cognitive_model_extraction_numbered.json
    Step 1 output: cognitive model per example, structured with keys:
    - "1. Relevant Story "
    - ...
    - "6. Automatic Thoughts "
    - "7. Emotions "
    - "8. Behaviors "

- outputs/refom_inference/(step2)_thought_path_generation_numbered.json
    Step 2 output: fine-grained thought paths, with fields:
    - id
    - fine-grained_thought_path
    - w_alphabet  (e.g., "(a) ...\n→ (b) ...\n→ (c) ...")
    - components  (list of step contents)

Outputs
-------
- (step2)_path_based_reasoning_raw.json
    Raw LLM outputs with soft yes/no scores and need_check flag.

- (step2)_path_based_reasoning_result.json
    Parsed results with:
    - distortion_assessment (yes/no)
    - need_check
    - distorted_hop
    - hop_text
    - causal_thought
    - result_thought

- (step2)_path_based_reasoning_eval_processed.json
    Flattened gold vs pred table for evaluation.

Prints:
    - Precision, Recall, F1 for binary distortion detection
    - Normal (No Distortion) recognition accuracy
"""

import os
import json
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from openai import OpenAI


# =========================
# Paths & configuration
# =========================

DATA_DIR = "data"
INFER_DIR = os.path.join("outputs", "reform_inference")
os.makedirs(INFER_DIR, exist_ok=True)

ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "Cognitive_Distortion_Detection.json")
COGNITIVE_MODEL_PATH = os.path.join(
    INFER_DIR, "(step1)_cognitive_model_extraction_numbered.json"
)
THOUGHT_PATH_PATH = os.path.join(
    INFER_DIR, "(step2)_thought_path_generation_numbered.json"
)

PATH_REASONING_RAW_PATH = os.path.join(
    INFER_DIR, "(step2)_path_based_reasoning_raw.json"
)
PATH_REASONING_RESULT_PATH = os.path.join(
    INFER_DIR, "(step2)_path_based_reasoning_result.json"
)
EVAL_PROCESSED_PATH = os.path.join(
    INFER_DIR, "(step2)_path_based_reasoning_eval_processed.json"
)

GENERATION_MODEL = "gpt-4o-mini-2024-07-18"
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.99  # threshold for "high confidence yes/no"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# =========================
# Utility functions
# =========================

def ensure_exists(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg} Missing file: {path}")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def letter_to_index_zero_based(letter: str) -> int:
    """
    Convert a step label letter to zero-based index.
    Example: 'a' -> 0, 'b' -> 1, ...
    """
    return ord(letter.lower()) - ord("a")


def map_gold_to_binary(label: str) -> int:
    """
    Gold label: 'No Distortion' -> 0, any other distortion -> 1.
    """
    return 0 if label == "No Distortion" else 1


def map_pred_to_binary(label: str) -> int:
    """
    Predicted label: 'no' -> 0, 'yes' -> 1 (case-insensitive).
    """
    return 0 if label.strip().lower() == "no" else 1


# =========================
# Step 1: Path-based Reasoning inference + soft scores
# =========================

def build_cognitive_model_string(cog_item: Dict[str, Any]) -> str:
    """
    Rebuild a flat cognitive model string from numbered fields.
    """
    fields = [
        "1. Relevant Story",
        "2. Core Beliefs",
        "3. Intermediate Beliefs",
        "4. Coping Strategies",
        "5. Situation",
        "6. Automatic Thoughts",
        "7. Emotions",
        "8. Behaviors",
    ]
    blocks: List[str] = []
    for key in fields:
        value = cog_item.get(key + " ", "Can't find")
        if value != "Can't find":
            blocks.append(f"{key}: {value}")
        else:
            blocks.append(f"{key}: (Not described)")
    return " ".join(blocks)


def build_path_based_prompt(
    cognitive_model_str: str,
    thought_path_text: str,
) -> str:
    """
    Build the path-based reasoning prompt for a single example.
    """
    return f"""You are a CBT therapist. Your goal is to determine whether there is any **cognitive distortion** present in the following **Thought Path**.

A **Thought Path** is a step-by-step reconstruction of the user's internal reasoning, from a situation to an automatic thought, emotion, or behavior. Your job is to evaluate whether any step contains **irrational leaps, exaggerated beliefs, or distorted logic**.

If a thought seems **plausible but exaggerated**, or shows **emotional overreaction**, err on the side of labeling it as **“yes”** for cognitive distortion.
If not, answer “no”.

## Input:
**Cognitive Model:**
{cognitive_model_str}

**Thought Path:**
{thought_path_text}

## Answer Format: (Copy the format exactly. Do not add any other text.)

1) yes or no  
2) hop where distortion begins (e.g., (b) → (c), or write NULL)  
"""


def compute_yes_no_soft_scores(logprobs_obj: Any) -> Dict[str, float]:
    """
    Aggregate token-level logprobs into approximate yes/no probabilities.

    The OpenAI Chat Completions logprobs API returns:
      choice.logprobs.content: list of token-level entries, each with
      .top_logprobs (list of {token, logprob, ...})

    We sum probabilities assigned to tokens 'yes' / 'no' (case-insensitive),
    then normalize to get a soft score over {yes, no}.
    """
    token_scores_raw: Dict[str, float] = {}

    content_items = getattr(logprobs_obj, "content", None)
    if content_items is None:
        return {"yes": 0.0, "no": 0.0}

    for entry in content_items:
        top_logprobs = getattr(entry, "top_logprobs", []) or []
        for tl in top_logprobs:
            token = tl.token.strip().lower()
            prob = math.exp(tl.logprob)
            token_scores_raw[token] = token_scores_raw.get(token, 0.0) + prob

    yes_prob = token_scores_raw.get("yes", 0.0)
    no_prob = token_scores_raw.get("no", 0.0)

    total = yes_prob + no_prob
    if total > 0:
        yes_prob /= total
        no_prob /= total

    # Round for readability
    return {"yes": round(yes_prob, 4), "no": round(no_prob, 4)}


def parse_predicted_label_from_answer(answer: str) -> str:
    """
    Parse the first line of the model answer to extract 'yes' or 'no'.

    Expected format:
        1) yes
        2) (b) → (c)
    """
    first_line = answer.split("\n")[0]  # e.g., '1) yes'
    if ")" in first_line:
        tail = first_line.split(")", 1)[1]
        return tail.strip().lower()
    return first_line.strip().lower()


def run_path_based_inference() -> List[Dict[str, Any]]:
    """
    Run path-based reasoning for each example and compute soft scores
    from logprobs. Returns a list of dictionaries with fields:

        - id
        - inferenced  (raw text answer)
        - soft_scores {'yes': p_yes, 'no': p_no}
        - need_check  ('yes' if we want to pass to Reverse Reasoning)
    """
    print("\n=== Step 1: Path-based reasoning + soft scores ===")

    ensure_exists(ORIGINAL_DATA_PATH, "[Step1]")
    ensure_exists(COGNITIVE_MODEL_PATH, "[Step1]")
    ensure_exists(THOUGHT_PATH_PATH, "[Step1]")

    original_data = load_json(ORIGINAL_DATA_PATH)
    cognitive_model_data = load_json(COGNITIVE_MODEL_PATH)
    thought_path_data = load_json(THOUGHT_PATH_PATH)

    assert len(original_data) == len(cognitive_model_data) == len(thought_path_data), \
        "Input lists must have the same length."

    results: List[Dict[str, Any]] = []

    for i in range(len(thought_path_data)):
        ex_id = original_data[i]["Id_Number"]
        cog_item = cognitive_model_data[i]
        thought_path_text = thought_path_data[i]["w_alphabet"]

        cog_str = build_cognitive_model_string(cog_item)
        prompt = build_path_based_prompt(cog_str, thought_path_text)

        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=0.3,
            logprobs=True,
            top_logprobs=5,
            messages=[{"role": "system", "content": prompt}],
        )

        choice = response.choices[0]
        answer_text = choice.message.content
        soft_scores = compute_yes_no_soft_scores(choice.logprobs)
        predicted_label = parse_predicted_label_from_answer(answer_text)

        # Confidence-based filtering:
        # We only "need_check" when the model predicts 'yes' but with low confidence.
        need_check = "no"
        if predicted_label == "yes":
            yes_conf = soft_scores.get("yes", 0.0)
            if yes_conf < CONFIDENCE_THRESHOLD:
                need_check = "yes"

        results.append(
            {
                "id": ex_id,
                "inferenced": answer_text,
                "soft_scores": soft_scores,
                "need_check": need_check,
            }
        )

        if (i + 1) % 50 == 0 or i == len(thought_path_data) - 1:
            print(f"  - processed {i+1}/{len(thought_path_data)} examples")

    save_json(results, PATH_REASONING_RAW_PATH)
    print(f"[Step1] Saved raw path-based reasoning outputs → {PATH_REASONING_RAW_PATH}")

    # Also return the data for downstream steps
    return results


# =========================
# Step 2: Map distorted hop to hop_text / causal/result thoughts
# =========================

def parse_answer_sections(answer: str) -> Dict[str, str]:
    """
    Parse '1) ... 2) ... 3) ...' style answer into a dictionary.

    We mainly need:
        1) yes/no
        2) distorted hop (e.g., '(b) → (c)' or 'NULL')
    """
    pattern = r"(1\)|2\)|3\))"
    parts = re.split(pattern, answer)
    # parts looks like: ["", "1)", " yes\n2)", " (b) → (c)\n3)", " ..."]
    if len(parts) < 3:
        return {}
    result = {}
    for idx in range(1, len(parts) - 1, 2):
        key = parts[idx].strip()
        value = parts[idx + 1].strip()
        result[key] = value
    return result


def build_path_reasoning_results(
    raw_results: List[Dict[str, Any]],
    thought_path_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine raw path-based reasoning outputs with Thought Path structure.

    For each example:
        - parse distortion assessment (yes/no)
        - parse distorted hop (e.g., '(b) → (c)')
        - reconstruct hop_text, causal_thought, result_thought
    """
    print("\n=== Step 2: Build hop-level info for Reverse Reasoning ===")

    # Index thought path by id
    path_by_id = {item["id"]: item for item in thought_path_data}

    results: List[Dict[str, Any]] = []

    for item in raw_results:
        ex_id = item["id"]
        answer = item["inferenced"]
        need_check = item["need_check"]

        sections = parse_answer_sections(answer)
        distortion_assessment = sections.get("1)", "no").strip().lower()
        distorted_hop = sections.get("2)", "NULL").strip()
        reason = sections.get("3)", "").strip()

        hop_text = "NULL"
        causal_thought = "NULL"
        result_thought = "NULL"

        if distorted_hop != "NULL" and ex_id in path_by_id:
            components = path_by_id[ex_id]["components"]

            # Case 1: (a) → (b) style reference
            match_pair = re.match(r"\((\w)\)\s*→\s*\((\w)\)", distorted_hop)
            if match_pair:
                start_letter = match_pair.group(1)
                end_letter = match_pair.group(2)
                s_idx = letter_to_index_zero_based(start_letter)
                e_idx = letter_to_index_zero_based(end_letter)

                if 0 <= s_idx < len(components) and 0 <= e_idx < len(components):
                    hop_text = (
                        f"({start_letter}) {components[s_idx]}\n"
                        f"→ ({end_letter}) {components[e_idx]}"
                    )
                    causal_thought = components[s_idx]
                    result_thought = components[e_idx]
            else:
                # Case 2: (x) → automatic thoughts
                match_auto = re.match(r"\((\w)\)\s*→\s*automatic thoughts", distorted_hop, re.IGNORECASE)
                if match_auto:
                    start_letter = match_auto.group(1)
                    s_idx = letter_to_index_zero_based(start_letter)
                    if 0 <= s_idx < len(components):
                        hop_text = (
                            f"({start_letter}) {components[s_idx]}\n"
                            f"→ automatic thoughts : {components[-1]}"
                        )
                        causal_thought = components[s_idx]
                        result_thought = components[-1]

        result_entry = {
            "id": ex_id,
            "distortion_assessment": distortion_assessment,  # 'yes' / 'no'
            "need_check": need_check,                        # 'yes' / 'no'
            "distorted_hop": distorted_hop,
            "reason": reason,
            "hop_text": hop_text,
            "causal_thought": causal_thought,
            "result_thought": result_thought,
        }
        results.append(result_entry)

    save_json(results, PATH_REASONING_RESULT_PATH)
    print(f"[Step2] Saved path-based reasoning results → {PATH_REASONING_RESULT_PATH}")
    return results


# =========================
# Step 3: Evaluation (binary detection)
# =========================

def evaluate_path_based_reasoning(
    path_results: List[Dict[str, Any]],
    original_data: List[Dict[str, Any]],
) -> None:
    """
    Evaluate binary distortion detection:

        Gold: Dominant Distortion (gold) vs "No Distortion"
        Pred: Distortion Assessment (pred) yes/no

    Metrics:
        - Precision, Recall, F1
        - Normal recognition accuracy (accuracy on gold == No Distortion)
    """
    print("\n=== Step 3: Evaluate path-based reasoning (binary detection) ===")

    assert len(path_results) == len(original_data), \
        "Evaluation assumes the same ordering and length."

    eval_rows: List[Dict[str, Any]] = []
    for i in range(len(path_results)):
        gold = original_data[i]
        pred = path_results[i]

        row = {
            "id": pred["id"],
            "Patient Question": gold["Patient Question"],
            "Distorted part (gold)": gold["Distorted part"],
            "Dominant Distortion (gold)": gold["Dominant Distortion"],
            "Secondary Distortion (gold)": gold["Secondary Distortion (Optional)"],
            "Distortion Assessment (pred)": pred["distortion_assessment"],
            "need_check": pred["need_check"],
        }
        eval_rows.append(row)

    save_json(eval_rows, EVAL_PROCESSED_PATH)
    print(f"[Step3] Saved evaluation table → {EVAL_PROCESSED_PATH}")

    df = pd.DataFrame(eval_rows)

    # Binary labels
    y_true = df["Dominant Distortion (gold)"].apply(map_gold_to_binary)
    y_pred = df["Distortion Assessment (pred)"].apply(map_pred_to_binary)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Normal (No Distortion) recognition accuracy
    no_dist_mask = df["Dominant Distortion (gold)"] == "No Distortion"
    no_dist_data = df[no_dist_mask]
    if len(no_dist_data) > 0:
        correct_no = (no_dist_data["Distortion Assessment (pred)"].str.lower() == "no").sum()
        normal_acc = correct_no / len(no_dist_data)
    else:
        normal_acc = 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Normal recognition accuracy (No Distortion → 'no'): {normal_acc:.4f}")


# =========================
# Main
# =========================

def main() -> None:
    # 1) Run path-based reasoning + soft scores (or reload if already done)
    if os.path.exists(PATH_REASONING_RAW_PATH):
        print("[Main] Found existing path-based reasoning outputs. Loading...")
        raw_results = load_json(PATH_REASONING_RAW_PATH)
    else:
        raw_results = run_path_based_inference()

    # 2) Build hop-level info for reverse reasoning
    thought_path_data = load_json(THOUGHT_PATH_PATH)
    path_results = build_path_reasoning_results(raw_results, thought_path_data)

    # 3) Evaluation (binary detection)
    original_data = load_json(ORIGINAL_DATA_PATH)
    evaluate_path_based_reasoning(path_results, original_data)

    # Quick summary: how many need_check == "yes"?
    num_need_check = sum(1 for r in path_results if r["need_check"] == "yes")
    print(f"\n[Summary] Number of examples flagged for Reverse Reasoning (need_check='yes'): {num_need_check}")


if __name__ == "__main__":
    main()
