"""
evaluate_distortion_classification.py

Evaluate the distortion-type classification model on the
Cognitive Distortion Detection dataset.

Inputs
------
- outputs/inference/distortion_classification_results.json
    [
      {
        "input_id": <example id>,
        "inferenced": "<raw model output text>"
      },
      ...
    ]

- data/raw/Cognitive_Distortion_Detection.json
    Original dataset with fields:
      - Id_Number
      - Dominant Distortion
      - Secondary Distortion (Optional)
      - ...

Outputs
-------
- Prints weighted Precision / Recall / F1 across all distortion labels
  (excluding "No Distortion" samples from gold).
"""

import os
import json
import re
from typing import List, Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report


# =========================
# Paths & label definitions
# =========================

DISTORTION_RESULTS_PATH = os.path.join(
    "outputs", "inference", "distortion_classification_results.json"
)
ORIGINAL_DATA_PATH = os.path.join(
    "data", "raw", "Cognitive_Distortion_Detection.json"
)

DISTORTION_LABELS = [
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental filter",
    "Should statements",
    "Labeling",
    "Personalization",
    "Magnification",
    "Emotional Reasoning",
    "Mind Reading",
    "Fortune-telling",
]


# =========================
# Text normalization & label pattern utilities
# =========================

def normalize_text(s: str) -> str:
    """
    Normalize text to make label detection more robust:
    - Remove 'Answer:' marker
    - Lowercase
    - Replace hyphens
    - Strip non-alphanumeric characters
    - Collapse multiple spaces
    """
    s = s.replace("Answer:", " ")
    s = s.lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # keep alphanumeric + spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_label_patterns(labels: List[str]) -> Dict[str, List[str]]:
    """
    For each canonical label, build a list of normalized patterns
    to search for inside model outputs.
    """
    patterns = {}
    for lbl in labels:
        base = normalize_text(lbl)
        patterns[lbl] = [base]

    # Common spelling/format variants
    patterns["Fortune-telling"].append("fortune telling")
    patterns["All-or-nothing thinking"].append("all or nothing")
    patterns["Overgeneralization"].append("overgeneral")  # truncated variant

    # If needed, you can add more variants here, e.g.:
    # patterns["Labeling"].append("labelling")
    # patterns["Personalization"].append("personalisation")

    return patterns


# =========================
# Multi-label helpers
# =========================

def to_indicator(instance_labels: List[str], label_vocab: List[str]) -> List[int]:
    """
    Convert a list of label strings into a binary indicator vector
    over a fixed label vocabulary.
    """
    label_set = set(instance_labels)
    return [1 if lbl in label_set else 0 for lbl in label_vocab]


# =========================
# Main evaluation pipeline
# =========================

def main():
    # 1) Load model outputs + gold data
    with open(DISTORTION_RESULTS_PATH, "r", encoding="utf-8") as f:
        distortion_classification_results = json.load(f)

    with open(ORIGINAL_DATA_PATH, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # 2) Parse raw model outputs into cleaned label lists
    label_patterns = build_label_patterns(DISTORTION_LABELS)
    processed_results = []

    for rec in distortion_classification_results:
        inferenced_text = rec.get("inferenced", "")
        norm = normalize_text(inferenced_text)

        # Collect label hit positions: (start_index, label)
        hits = []
        for label, patterns in label_patterns.items():
            best_pos = None
            for pat in patterns:
                idx = norm.find(pat)
                if idx != -1:
                    best_pos = idx if best_pos is None else min(best_pos, idx)
            if best_pos is not None:
                hits.append((best_pos, label))

        # Sort by first occurrence and deduplicate
        hits.sort(key=lambda x: x[0])
        ordered_labels = []
        seen = set()
        for _, lbl in hits:
            if lbl not in seen:
                seen.add(lbl)
                ordered_labels.append(lbl)

        # Use at most the first two labels
        cleaned_labels = ordered_labels[:2]

        processed_results.append(
            {
                "id": rec["input_id"],
                "distortion_inferenced": cleaned_labels,
            }
        )

    # 3) Build id -> prediction map
    pred_map = {}
    for row in processed_results:
        ex_id = str(row.get("id"))
        preds = row.get("distortion_inferenced", []) or []
        preds = [p.strip() for p in preds if isinstance(p, str) and p.strip()]
        pred_map[ex_id] = preds

    # 4) Build gold label list for evaluation
    #    Only examples whose Dominant Distortion != "No Distortion" are evaluated.
    eval_gold = []  # elements: {"id": str, "gold_labels": List[str]}
    for row in original_data:
        dominant = (row.get("Dominant Distortion") or "").strip()
        secondary = (row.get("Secondary Distortion (Optional)") or "").strip()

        if not dominant or dominant == "No Distortion":
            # Skip non-distorted posts for classification evaluation
            continue

        gold_labels = [dominant]

        if secondary:
            sec_labels = [s.strip() for s in secondary.split(",") if s.strip()]
            for s in sec_labels:
                if s != "No Distortion":
                    gold_labels.append(s)

        gold_labels = sorted(set(gold_labels))
        eval_gold.append(
            {
                "id": str(row.get("Id_Number")),
                "gold_labels": gold_labels,
            }
        )

    # 5) Define label space (we use DISTORTION_LABELS in given order)
    labels = [lbl for lbl in DISTORTION_LABELS]

    # 6) Convert gold/pred to multi-label indicator format
    Y_true = []
    Y_pred = []
    missing_pred_ids = []

    for eg in eval_gold:
        ex_id = eg["id"]
        gold_labels = eg["gold_labels"]

        Y_true.append(to_indicator(gold_labels, labels))

        preds = pred_map.get(ex_id, [])
        if ex_id not in pred_map:
            missing_pred_ids.append(ex_id)
        Y_pred.append(to_indicator(preds, labels))

    Y_true = np.array(Y_true, dtype=int)
    Y_pred = np.array(Y_pred, dtype=int)

    # 7) Compute weighted Precision / Recall / F1 over labels
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="weighted", zero_division=0
    )

    print(f"Number of evaluated gold samples (distorted posts): {len(eval_gold)}")
    print(f"Number of samples with missing prediction (treated as empty): {len(missing_pred_ids)}\n")

    print(f"Weighted Precision: {precision_w:.4f}")
    print(f"Weighted Recall   : {recall_w:.4f}")
    print(f"Weighted F1       : {f1_w:.4f}")

    # Optional: per-label report
    # print("\n[Per-label classification report]")
    # print(classification_report(Y_true, Y_pred, target_names=labels, zero_division=0))


if __name__ == "__main__":
    main()
