"""
Update_initial_Assessment_and_Assessment_Evaluation.py

Step 3: Update the initial path-based distortion assessment using
the Overpathologizing Detection Model (ODM), 
then evaluate the updated binary distortion detection performance.

Inputs
------
1) Path-based reasoning results
   - outputs/path_based_reasoning/path_based_reasoning_results.json
   Each item is expected to have at least:
     {
       "id": <example_id>,
       "distortion_assessment": "yes" or "no",
       ...
     }

2) ODM predictions
   - outputs/reverse_reasoning/odm_predictions.json
   Each item is expected to have:
     {
       "input_id": <example_id>,   # same as 'id' in path-based results
       "inferenced": "<model free-form answer containing 'reasonable' or 'unreasonable'>",
       ...
     }

3) Original dataset (for evaluation)
   - data/Cognitive Distortion Detection.json
   Each item has:
     {
       "Id_Number": ...,
       "Dominant Distortion": "No Distortion" or <distortion type>,
       ...
     }

Outputs
-------
1) Updated distortion assessment per example:
   - outputs/reverse_reasoning/updated_assessment_results.json
     [
       {"id": ..., "distortion_assessment": "yes" or "no"},
       ...
     ]

2) Printed evaluation metrics:
   - Precision / Recall / F1
   - Specificity for the "No Distortion" class
"""

import os
import json
from typing import List, Dict

from sklearn.metrics import precision_score, recall_score, f1_score


# =========================
# Paths
# =========================

DATA_DIR = "data"
OUTPUT_DIR_REASONING = os.path.join("outputs", "path_based_reasoning")
OUTPUT_DIR_REVERSE = os.path.join("outputs", "reverse_reasoning")

PATH_REASONING_RESULTS = os.path.join(
    OUTPUT_DIR_REASONING, "path_based_reasoning_results.json"
)
ODM_PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR_REVERSE, "odm_predictions.json"
)
UPDATED_ASSESSMENT_PATH = os.path.join(
    OUTPUT_DIR_REVERSE, "updated_assessment_results.json"
)

ORIGINAL_DATASET_PATH = os.path.join(DATA_DIR, "Cognitive Distortion Detection.json")


# =========================
# Loading helpers
# =========================

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 1) Update assessment
# =========================

def update_assessment_with_odm(
    path_reasoning_results: List[Dict],
    odm_predictions: List[Dict],
) -> List[Dict]:
    """
    Update the binary distortion assessment ("yes"/"no") using ODM outputs.

    Logic (current version):
    - If ODM output text contains "unreasonable"  → distortion_assessment = "yes"
    - If ODM output text contains "reasonable"    → distortion_assessment = "no"
    - Otherwise, fall back to the original path-based assessment.

    Notes
    -----
    - This assumes the fine-tuned ODM outputs the words "reasonable" / "unreasonable".
      If your model uses "overpathologized" / "not overpathologized" instead,
      adjust the mapping below accordingly.
    """

    # Map from example_id → ODM prediction item
    odm_by_id = {str(item["input_id"]): item for item in odm_predictions}

    updated_result = []

    for result in path_reasoning_results:
        example_id = str(result["id"])
        current_assessment = result.get("distortion_assessment", "no")

        matched = odm_by_id.get(example_id, None)

        if matched is not None:
            text = str(matched.get("inferenced", "")).lower()

            if "unreasonable" in text:
                new_assessment = "yes"
            elif "reasonable" in text:
                new_assessment = "no"
            else:
                # If ODM output is unclear, keep original label
                new_assessment = current_assessment
        else:
            # If no ODM prediction is available, keep original label
            new_assessment = current_assessment

        updated_result.append(
            {
                "id": example_id,
                "distortion_assessment": new_assessment,
            }
        )

    return updated_result


# =========================
# 2) Evaluation
# =========================

def build_gold_labels(original_data: List[Dict]) -> Dict[str, str]:
    """
    Build gold binary labels from the original dataset.
    - "No Distortion" → "no"
    - any other Dominant Distortion → "yes"
    """
    id_to_gold = {}
    for item in original_data:
        example_id = str(item["Id_Number"])
        if item.get("Dominant Distortion", "No Distortion") != "No Distortion":
            label = "yes"
        else:
            label = "no"
        id_to_gold[example_id] = label
    return id_to_gold


def build_pred_labels(updated_result: List[Dict]) -> Dict[str, str]:
    """
    Build predicted labels dict: example_id → "yes"/"no"
    """
    id_to_pred = {}
    for item in updated_result:
        example_id = str(item["id"])
        predicted_label = item.get("distortion_assessment", "no")
        id_to_pred[example_id] = predicted_label
    return id_to_pred


def evaluate_binary_detection(
    id_to_gold: Dict[str, str],
    id_to_pred: Dict[str, str],
) -> None:
    """
    Evaluate binary distortion detection performance:
    - Precision / Recall / F1 on the "yes" class
    - Specificity: P(pred=no | gold=no)
    """
    common_ids = sorted(set(id_to_gold.keys()) & set(id_to_pred.keys()))

    y_true = [id_to_gold[i] for i in common_ids]
    y_pred = [id_to_pred[i] for i in common_ids]

    precision = precision_score(y_true, y_pred, pos_label="yes")
    recall = recall_score(y_true, y_pred, pos_label="yes")
    f1 = f1_score(y_true, y_pred, pos_label="yes")

    print(f"Precision (label='yes'): {precision:.4f}")
    print(f"Recall    (label='yes'): {recall:.4f}")
    print(f"F1 Score  (label='yes'): {f1:.4f}")

    # Specificity for the "No Distortion" class
    true_negatives = sum(
        1 for i in common_ids
        if id_to_gold[i] == "no" and id_to_pred[i] == "no"
    )
    total_negatives = sum(1 for i in common_ids if id_to_gold[i] == "no")
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0.0

    print(
        f"Specificity (gold='no', pred='no'): {specificity:.4f}"
    )


# =========================
# Main
# =========================

def main():
    # 1) Load inputs
    path_reasoning_results = load_json(PATH_REASONING_RESULTS)
    odm_predictions = load_json(ODM_PREDICTIONS_PATH)
    original_data = load_json(ORIGINAL_DATASET_PATH)

    # 2) Update assessment using ODM
    updated_result = update_assessment_with_odm(
        path_reasoning_results=path_reasoning_results,
        odm_predictions=odm_predictions,
    )

    os.makedirs(OUTPUT_DIR_REVERSE, exist_ok=True)
    with open(UPDATED_ASSESSMENT_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_result, f, ensure_ascii=False, indent=4)

    print(f"Updated assessment saved to: {UPDATED_ASSESSMENT_PATH}")

    # 3) Build gold & predicted labels
    id_to_gold = build_gold_labels(original_data)
    id_to_pred = build_pred_labels(updated_result)

    # 4) Evaluate
    evaluate_binary_detection(id_to_gold, id_to_pred)


if __name__ == "__main__":
    main()
