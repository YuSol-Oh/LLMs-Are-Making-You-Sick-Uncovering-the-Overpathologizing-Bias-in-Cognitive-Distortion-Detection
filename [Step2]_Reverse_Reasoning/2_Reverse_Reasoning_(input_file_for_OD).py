"""
2_Reverse_Reasoning_input_for_OD.py

Step 2 in REFORM: Build input prompts for the Overpathologizing Detector (OD)
based on Reverse Reasoning.

This script takes:
    1) Path-based Reasoning results (with hop-level distortion info)
    2) Hop-level Thought Path Data (Thought_Path_Data)
    3) Original cognitive model (to recover the situation)

and produces:
    - A set of rewritten, situation-adapted causal thoughts
      for each "uncertain" distorted hop (need_check == "yes").
    - Instruction-style prompts that will be fed into the OD model.

Pipeline
--------
1) Load path-based reasoning results from Step 1 and filter examples
   with need_check == "yes".

2) Compute embeddings for all result_thoughts in Thought_Path_Data
   (or load precomputed embeddings if available).

3) For each target hop (from need_check):
   - Embed its result_thought.
   - Retrieve top-k (default: 5) most similar result_thought examples
     from Thought_Path_Data, excluding the same id.
   - Collect their causal_thoughts as retrieved candidates.

4) Ask an LLM to rewrite each retrieved causal thought so that it fits
   the target (seed) situation, producing five plausible causal thoughts
   for the given result_thought and situation.

5) Wrap:
    - target result_thought,
    - generated 5 causal thoughts,
    - original candidate causal thought
   into an instruction prompt that will serve as input to the OD model.

Inputs
------
- outputs/reform_inference/(step2)_path_based_reasoning_result.json
    From Step 1 (Path-based Reasoning + Confidence-based Filtering),
    entries with fields:
        - id
        - distorted_hop
        - hop_text
        - causal_thought
        - result_thought
        - situation (not included originally; re-added here via cognitive model)
        - need_check  ('yes' for Reverse Reasoning)

- outputs/data_augmentation/extracted_cognitive_model_numbered.json
    Cognitive model per example, with keys including:
        - "id"
        - "5. Situation "

- outputs/data_augmentation/Thought_Path_Data.json
    Hop-level Thought Path data (Thought Path Data), with fields:
        - id
        - hop_num
        - hop_text
        - causal_thought
        - result_thought
        - cognitive_model[ ... ]

Outputs
-------
- outputs/reverse_reasoning/embedded_thought_path_data.json
    Result-thought embeddings for all hops.

- outputs/reverse_reasoning/retrieved_for_input.json
    For each input hop that needs checking:
        - input_id
        - distorted_hop
        - input_causal_thought
        - input_result_thought
        - situation
        - top_5_retrieved (list of "id-hop_num")

- outputs/reverse_reasoning/input_retrieved_augmented.json
    For each input:
        - input_id
        - input_causal_thought
        - input_result_thought
        - generated_causal_thought (list of 5 rewritten causal thoughts)

- outputs/reverse_reasoning/input_for_OD.json
    Final OD prompts:
        - input_id
        - input_causal_thought
        - input_result_thought
        - input_prompt (instruction string)
"""

import os
import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from sentence_transformers import util


# =========================
# Paths & configuration
# =========================

DATA_DIR = "data"
AUG_DIR = os.path.join("outputs", "data_augmentation")
REFORM_DIR = os.path.join("outputs", "reform_inference")
RR_DIR = os.path.join("outputs", "reverse_reasoning")

os.makedirs(AUG_DIR, exist_ok=True)
os.makedirs(REFORM_DIR, exist_ok=True)
os.makedirs(RR_DIR, exist_ok=True)

# Inputs
PATH_REASONING_RESULT_PATH = os.path.join(
    REFORM_DIR, "(step2)_path_based_reasoning_result.json"
)
COGNITIVE_MODEL_PATH = os.path.join(
    AUG_DIR, "extracted_cognitive_model_numbered.json"
)
THOUGHT_PATH_DATA_PATH = os.path.join(
    AUG_DIR, "Thought_Path_Data.json"
)

# Intermediate / Outputs
EMBEDDED_THOUGHT_PATH_PATH = os.path.join(
    RR_DIR, "embedded_thought_path_data.json"
)
RETRIEVED_FOR_INPUT_PATH = os.path.join(
    RR_DIR, "retrieved_for_input.json"
)
AUGMENTED_REWRITTEN_PATH = os.path.join(
    RR_DIR, "input_retrieved_augmented.json"
)
OD_INPUT_PATH = os.path.join(
    RR_DIR, "input_for_OD.json"
)

GENERATION_MODEL = "gpt-4o-mini-2024-07-18"
EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K = 5

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


def embed_text(text: str) -> List[float]:
    """
    Get an embedding for a single text using the OpenAI embeddings API.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


# =========================
# Step 1: Collect examples that need Reverse Reasoning
# =========================

def collect_need_check_examples() -> List[Dict[str, Any]]:
    """
    Load path-based reasoning results and cognitive models,
    then build a list of examples where need_check == 'yes'.

    For each example, we keep:
        - id
        - distorted_hop
        - hop_text
        - causal_thought
        - result_thought
        - situation
    """
    print("\n=== Step 1: Collect examples flagged for Reverse Reasoning ===")

    ensure_exists(PATH_REASONING_RESULT_PATH, "[RR Step1]")
    ensure_exists(COGNITIVE_MODEL_PATH, "[RR Step1]")

    reasoning_path_results = load_json(PATH_REASONING_RESULT_PATH)
    cognitive_model_data = load_json(COGNITIVE_MODEL_PATH)

    cog_by_id = {item["id"]: item for item in cognitive_model_data}

    need_check: List[Dict[str, Any]] = []

    for result in reasoning_path_results:
        if result.get("need_check") != "yes":
            continue

        ex_id = result["id"]
        cog_item = cog_by_id.get(ex_id, {})
        situation = cog_item.get("5. Situation ", "")

        causal_thought = result.get("causal_thought", "").strip()
        result_thought = result.get("result_thought", "").strip()

        need_check.append(
            {
                "id": ex_id,
                "distorted_hop": result.get("distorted_hop", ""),
                "hop_text": result.get("hop_text", ""),
                "causal_thought": causal_thought,
                "result_thought": result_thought,
                "situation": situation,
            }
        )

    print(f"[RR Step1] Collected {len(need_check)} examples with need_check='yes'.")
    return need_check


# =========================
# Step 2: Embed all Thought Path result_thoughts
# =========================

def build_or_load_embedded_thought_path_data() -> List[Dict[str, Any]]:
    """
    Build (or load) embeddings for result_thought in Thought_Path_Data.

    Output format (list of dicts):
        - id
        - hop_num
        - result_thought
        - embedded_result_thought
    """
    print("\n=== Step 2: Build / load embeddings for Thought_Path_Data ===")

    ensure_exists(THOUGHT_PATH_DATA_PATH, "[RR Step2]")

    if os.path.exists(EMBEDDED_THOUGHT_PATH_PATH):
        print(f"[RR Step2] Found existing embedding file. Loading: {EMBEDDED_THOUGHT_PATH_PATH}")
        return load_json(EMBEDDED_THOUGHT_PATH_PATH)

    thought_path_data = load_json(THOUGHT_PATH_DATA_PATH)
    embedded_data: List[Dict[str, Any]] = []

    for idx, item in enumerate(thought_path_data):
        result_thought = item.get("result_thought", "")
        embedding = embed_text(result_thought)

        embedded_data.append(
            {
                "id": item["id"],
                "hop_num": item["hop_num"],
                "result_thought": result_thought,
                "embedded_result_thought": embedding,
            }
        )

        if (idx + 1) % 100 == 0 or idx == len(thought_path_data) - 1:
            print(f"  - embedded {idx+1}/{len(thought_path_data)} hops")

    save_json(embedded_data, EMBEDDED_THOUGHT_PATH_PATH)
    print(f"[RR Step2] Saved embedded Thought Path data → {EMBEDDED_THOUGHT_PATH_PATH}")
    return embedded_data


# =========================
# Step 3: Retrieve top-k similar hops per target (result_thought)
# =========================

def retrieve_similar_examples_for_need_check(
    need_check: List[Dict[str, Any]],
    embedded_thought_path_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each example in need_check, retrieve top-k most similar result_thought
    from the full Thought_Path_Data (excluding the same id).

    Returns a list of dicts:
        - input_id
        - distorted_hop
        - input_causal_thought
        - input_result_thought
        - situation
        - top_5_retrieved (list of "id-hop_num" strings)
    """
    print("\n=== Step 3: Retrieve top-k similar result_thought examples ===")

    input_with_retrieved: List[Dict[str, Any]] = []

    for idx, ex in enumerate(need_check):
        ex_id = ex["id"]
        ex_result = ex["result_thought"]

        ex_embedding = embed_text(ex_result)

        retrieve_scores: List[Tuple[float, str, int]] = []
        for retrieve_item in embedded_thought_path_data:
            if retrieve_item["id"] == ex_id:
                continue

            retrieve_embedding = retrieve_item["embedded_result_thought"]
            # util.cos_sim expects tensors, but it also works on lists via conversion
            similarity = util.cos_sim(ex_embedding, retrieve_embedding).item()
            retrieve_scores.append(
                (similarity, retrieve_item["id"], retrieve_item["hop_num"])
            )

        # sort by similarity (desc) and take top-k
        top_k_items = sorted(retrieve_scores, key=lambda x: x[0], reverse=True)[:TOP_K]

        input_with_retrieved.append(
            {
                "input_id": ex_id,
                "distorted_hop": ex["distorted_hop"],
                "input_causal_thought": ex["causal_thought"],
                "input_result_thought": ex["result_thought"],
                "situation": ex["situation"],
                "top_5_retrieved": [f"{item[1]}-{item[2]}" for item in top_k_items],
            }
        )

        if (idx + 1) % 50 == 0 or idx == len(need_check) - 1:
            print(f"  - retrieved neighbors for {idx+1}/{len(need_check)} examples")

    save_json(input_with_retrieved, RETRIEVED_FOR_INPUT_PATH)
    print(f"[RR Step3] Saved retrieved neighbors → {RETRIEVED_FOR_INPUT_PATH}")
    return input_with_retrieved


# =========================
# Step 4: Rewrite retrieved causal thoughts to match the seed situation
# =========================

def build_retrieved_causal_thought_struct(
    input_with_retrieved: List[Dict[str, Any]],
    thought_path_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each input example, expand its top_5_retrieved into actual
    retrieved hop objects, including their causal_thought.

    Returns list of dicts:
        - input_id
        - input_causal_thought
        - input_result_thought
        - situation
        - top_5_retrieved
        - retrieved_examples (list of Thought_Path_Data entries)
    """
    print("\n=== Step 4: Attach retrieved hop objects (including causal_thought) ===")

    # Index Thought_Path_Data by (id, hop_num)
    path_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for item in thought_path_data:
        key = (item["id"], item["hop_num"])
        path_index[key] = item

    retrieved_causal_thought: List[Dict[str, Any]] = []

    for ex in input_with_retrieved:
        retrieved_examples: List[Dict[str, Any]] = []

        for r in ex["top_5_retrieved"]:
            example_id, hop_str = r.split("-")
            hop_num = int(hop_str)

            matched_example = path_index.get((example_id, hop_num))
            # We expect all to match; if not, keep None to signal an issue.
            retrieved_examples.append(matched_example)

        retrieved_causal_thought.append(
            {
                "input_id": ex["input_id"],
                "input_causal_thought": ex["input_causal_thought"],
                "input_result_thought": ex["input_result_thought"],
                "situation": ex["situation"],
                "top_5_retrieved": ex["top_5_retrieved"],
                "retrieved_examples": retrieved_examples,
            }
        )

    print(f"[RR Step4] Built retrieved_causal_thought entries: {len(retrieved_causal_thought)}")
    return retrieved_causal_thought


def build_rewrite_prompt(
    situation: str,
    retrieved_causal_thoughts: List[str],
) -> str:
    """
    Build the prompt to rewrite retrieved causal thoughts so that
    they are adapted to the given seed situation.
    """
    return f"""You are a reasoning-oriented agent specialized in cognitive modeling.  
Your task is to generate **five distinct, plausible, and contextually adapted causal thoughts** based on a **seed situation** and **five retrieved thoughts**.

## Objective:
You are given five **retrieved causal thoughts** from different contexts.

Your task is to **rewrite each of the five retrieved thoughts** so that they fit the **seed situation**.

## How to proceed:

* Adapt each retrieved causal thought **independently** to match the **seed situation**.  
* Do not alter the structure or psychological tone too much from the original retrieved causal thought.  
* The result should **read as a psychologically natural and situationally coherent thought** that someone might plausibly have **in the given seed situation**.

## Input:

* **Retrieved causal thoughts** (from similar patterns):
  1. {retrieved_causal_thoughts[0]}
  2. {retrieved_causal_thoughts[1]}
  3. {retrieved_causal_thoughts[2]}
  4. {retrieved_causal_thoughts[3]}
  5. {retrieved_causal_thoughts[4]}

* **Seed situation** (what actually happened):
  {situation}

## Output Format:

You must rewrite **each** retrieved causal thought to match the seed situation.
Write all five outputs in the following format:

1. (rewritten version of retrieved thought 1 adapted to the seed situation)
2. (rewritten version of retrieved thought 2 adapted to the seed situation)
3. (rewritten version of retrieved thought 3 adapted to the seed situation)
4. (rewritten version of retrieved thought 4 adapted to the seed situation)
5. (rewritten version of retrieved thought 5 adapted to the seed situation)

**Each output should be a single sentence.**
**Do not include any explanation, notes, or bracketed text.**
**You must output all five.**
"""


def rewrite_retrieved_causal_thoughts(
    retrieved_causal_thought: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    For each input example, call the LLM to rewrite the top-5 retrieved
    causal thoughts so that they fit the seed situation.

    Returns list of dicts:
        - input_id
        - input_causal_thought
        - input_result_thought
        - generated_causal_thought (list of 5 rewritten causal thoughts)
    """
    print("\n=== Step 5: Rewrite retrieved causal thoughts with LLM ===")

    augmented_inferenced: List[Dict[str, Any]] = []

    for idx, ex in enumerate(retrieved_causal_thought):
        # Skip if any retrieved example is missing
        if any(item is None for item in ex["retrieved_examples"]):
            continue

        retrieved_examples_causal = [
            item["causal_thought"] for item in ex["retrieved_examples"]
        ]

        prompt = build_rewrite_prompt(
            situation=ex["situation"],
            retrieved_causal_thoughts=retrieved_examples_causal,
        )

        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=0.3,
            messages=[{"role": "system", "content": prompt}],
        )
        answer = response.choices[0].message.content

        # Parse enumerated outputs: "1. ...\n2. ..."
        split_items = re.split(r"\n?\s*\d+\.\s+", answer.strip())
        # split_items[0] is empty or prefix, 1:~ are the five lines
        generated = split_items[1:]

        augmented_inferenced.append(
            {
                "input_id": ex["input_id"],
                "input_causal_thought": ex["input_causal_thought"],
                "input_result_thought": ex["input_result_thought"],
                "generated_causal_thought": generated,
            }
        )

        if (idx + 1) % 20 == 0 or idx == len(retrieved_causal_thought) - 1:
            print(f"  - rewritten {idx+1}/{len(retrieved_causal_thought)} examples")

    save_json(augmented_inferenced, AUGMENTED_REWRITTEN_PATH)
    print(f"[RR Step5] Saved rewritten causal thoughts → {AUGMENTED_REWRITTEN_PATH}")
    return augmented_inferenced


# =========================
# Step 5: Build OD input prompts
# =========================

def build_od_input(
    augmented_inferenced: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build instruction-style prompts for the Overpathologizing Detector (OD).

    Each prompt describes:
        - Target Thought (result)
        - Plausible Causal Thoughts (5)
        - Candidate Causal Thought to evaluate

    OD is expected to answer whether the candidate causal thought is
    reasonably included among plausible causes (model will later map this
    to 'overpathologized' / 'not overpathologized').
    """
    print("\n=== Step 6: Build input prompts for OD ===")

    od_inputs: List[Dict[str, Any]] = []

    for item in augmented_inferenced:
        instruction = f"""You will be given the following:
- A 'target thought' (the resulting thought),
- A set of 5 'plausible causal thoughts' that could reasonably lead to the target thought,
- And 1 additional 'candidate causal thought' whose validity you need to evaluate.

Determine whether the 'candidate causal thought' is reasonably included among the plausible causal thoughts that could give rise to the target thought.

Target Thought (result): {item['input_result_thought']}
Plausible Causal Thoughts (5): {item['generated_causal_thought']}
Candidate Causal Thought to Evaluate: {item['input_causal_thought']}

Answer:
"""

        od_inputs.append(
            {
                "input_id": item["input_id"],
                "input_causal_thought": item["input_causal_thought"],
                "input_result_thought": item["input_result_thought"],
                "input_prompt": instruction,
            }
        )

    save_json(od_inputs, OD_INPUT_PATH)
    print(f"[RR Step6] Saved OD input prompts → {OD_INPUT_PATH}")
    return od_inputs


# =========================
# Main
# =========================

def main() -> None:
    # Step 1: Examples that need Reverse Reasoning
    need_check_examples = collect_need_check_examples()
    if not need_check_examples:
        print("[Main] No examples with need_check='yes'. Nothing to do.")
        return

    # Step 2: Embeddings for all Thought_Path_Data
    embedded_thought_path_data = build_or_load_embedded_thought_path_data()

    # Step 3: Retrieve top-k similar examples
    input_with_retrieved = retrieve_similar_examples_for_need_check(
        need_check_examples,
        embedded_thought_path_data,
    )

    # Step 4: Attach retrieved hop objects (with causal_thought)
    thought_path_data = load_json(THOUGHT_PATH_DATA_PATH)
    retrieved_causal_thought = build_retrieved_causal_thought_struct(
        input_with_retrieved,
        thought_path_data,
    )

    # Step 5: Rewrite retrieved causal thoughts to match the seed situation
    augmented_inferenced = rewrite_retrieved_causal_thoughts(retrieved_causal_thought)

    # Step 6: Build OD input prompts
    build_od_input(augmented_inferenced)

    print("\n[Main] Reverse Reasoning input file for OD has been generated.")


if __name__ == "__main__":
    main()
