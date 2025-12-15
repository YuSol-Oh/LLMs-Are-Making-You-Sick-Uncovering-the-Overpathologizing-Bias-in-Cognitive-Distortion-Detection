"""
1_Overpathologize_Detector_Model_dataset_generation.py

Builds an instruction fine-tuning dataset for the Overpathologizing Detector.

Pipeline overview
-----------------
Input:
    - Thought_Path_Data_ver03.json
      (hop-level dataset with fields such as:
       - id, hop_num, hop_text, distortion ("No Distortion"/"Yes Distortion"),
       - causal_thought, result_thought, cognitive_model["5. Situation "])

Steps:
    1) Split hops into:
       - reasonable_seed_base (No Distortion hops)
       - unreasonable_seed_base (Yes Distortion hops)
       - retrieve_base (shared No Distortion pool)

    2) Compute OpenAI embeddings for the result_thought of:
       - reasonable_seed_base
       - unreasonable_seed_base
       - retrieve_base

    3) For each seed hop, retrieve top-k (k=5) similar hops from retrieve_base
       based on cosine similarity of the embeddings.

    4) For each seed (reasonable/unreasonable separately), build
       (seed, retrieved_examples) tuples and call an LLM to generate
       5 adapted "causal thoughts" for the seed situation.

    5) From generated thoughts, construct an instruction-style fine-tuning dataset:
       - For reasonable seeds: label = "overpathologized"
       - For unreasonable seeds: label = "not overpathologized"

    6) Split into train / valid / test sets and save to JSON.

Notes:
    - This script uses synchronous Chat Completions for clarity.
      You can adapt the generation part to the OpenAI Batch API if desired.
    - Requires OPENAI_API_KEY to be set in the environment.
"""

import os
import json
import random
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from openai import OpenAI


# =========================
# Paths & configuration
# =========================

DATA_DIR = "data"
OUTPUT_DIR = os.path.join("outputs", "reverse_reasoning", "ver03")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THOUGHT_PATH_DATA_PATH = os.path.join(
    DATA_DIR, "Thought_Path_Data_ver03.json"
)

# Seed / retrieve base outputs
REASONABLE_SEED_BASE_PATH = os.path.join(
    OUTPUT_DIR, "reasonable_seed_base.json"
)
UNREASONABLE_SEED_BASE_PATH = os.path.join(
    OUTPUT_DIR, "unreasonable_seed_base.json"
)
RETRIEVE_BASE_PATH = os.path.join(
    OUTPUT_DIR, "retrieve_base.json"
)

# Embedded versions
REASONABLE_SEED_EMB_PATH = os.path.join(
    OUTPUT_DIR, "reasonable_seed_base_embedded.json"
)
UNREASONABLE_SEED_EMB_PATH = os.path.join(
    OUTPUT_DIR, "unreasonable_seed_base_embedded.json"
)
RETRIEVE_BASE_EMB_PATH = os.path.join(
    OUTPUT_DIR, "retrieve_base_embedded.json"
)

# Retrieval outputs
REASONABLE_RETRIEVED_TOPK_PATH = os.path.join(
    OUTPUT_DIR, "reasonable_retrieved_top_5_examples.json"
)
UNREASONABLE_RETRIEVED_TOPK_PATH = os.path.join(
    OUTPUT_DIR, "unreasonable_retrieved_top_5_examples.json"
)

# Augmentation outputs (LLM generated causal thoughts)
REASONABLE_AUGMENTED_PATH = os.path.join(
    OUTPUT_DIR, "reasonable_data_augmentation.json"
)
UNREASONABLE_AUGMENTED_PATH = os.path.join(
    OUTPUT_DIR, "unreasonable_data_augmentation.json"
)

# Final fine-tuning datasets
FT_ALL_PATH = os.path.join(
    OUTPUT_DIR, "finetuning_data_overpathologizing_detection_all.json"
)
FT_TRAIN_PATH = os.path.join(
    OUTPUT_DIR, "finetuning_data_overpathologizing_detection_train.json"
)
FT_VALID_PATH = os.path.join(
    OUTPUT_DIR, "finetuning_data_overpathologizing_detection_valid.json"
)
FT_TEST_PATH = os.path.join(
    OUTPUT_DIR, "finetuning_data_overpathologizing_detection_test.json"
)

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATION_MODEL = "gpt-4o-mini-2024-07-18"

RANDOM_SEED = 42
TOP_K = 5

# OpenAI client (expects OPENAI_API_KEY in the environment)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# =========================
# Utility functions
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def ensure_exists(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg} Missing file: {path}")


def get_embedding(text: str) -> List[float]:
    """
    Get a single text embedding using OpenAI embeddings API.
    """
    resp = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return resp.data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two embedding vectors (list of floats).
    """
    v1 = np.array(a, dtype=np.float32)
    v2 = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# =========================
# Step 1: Split seed & retrieve bases
# =========================

def split_seed_and_retrieve(thought_path_data: List[Dict[str, Any]]) -> None:
    """
    Split Thought Path data into:
        - reasonable_seed_base: hops with distortion == "No Distortion"
        - unreasonable_seed_base: hops with distortion != "No Distortion"
        - retrieve_base: remaining No Distortion hops (second half)

    Uses fixed random seed for reproducibility.
    """
    print("\n=== Step 1: Split seed and retrieve bases ===")

    random.seed(RANDOM_SEED)

    no_distortion_data = [
        item for item in thought_path_data
        if item.get("distortion", "") == "No Distortion"
    ]
    yes_distortion_data = [
        item for item in thought_path_data
        if item.get("distortion", "") != "No Distortion"
    ]

    random.shuffle(no_distortion_data)

    split_idx = len(no_distortion_data) // 2
    reasonable_seed_base = no_distortion_data[:split_idx]
    retrieve_base = no_distortion_data[split_idx:]
    unreasonable_seed_base = yes_distortion_data

    print(f"Reasonable Seed Base:  {len(reasonable_seed_base)}")
    print(f"Unreasonable Seed Base:{len(unreasonable_seed_base)}")
    print(f"Shared Retrieve Base:  {len(retrieve_base)}")

    save_json(reasonable_seed_base, REASONABLE_SEED_BASE_PATH)
    save_json(unreasonable_seed_base, UNREASONABLE_SEED_BASE_PATH)
    save_json(retrieve_base, RETRIEVE_BASE_PATH)

    print(f"[Step1] Saved bases to:\n"
          f"  {REASONABLE_SEED_BASE_PATH}\n"
          f"  {UNREASONABLE_SEED_BASE_PATH}\n"
          f"  {RETRIEVE_BASE_PATH}")


# =========================
# Step 2: Compute embeddings
# =========================

def embed_base(
    base_data: List[Dict[str, Any]],
    out_path: str,
    base_name: str,
) -> None:
    """
    Compute embeddings for result_thought in a base (seed or retrieve) and save.

    Each entry in the output will contain:
        - id
        - hop_num
        - hop_text
        - result_thought
        - situation (from cognitive_model["5. Situation "])
        - result_embedding
    """
    print(f"\n[Step2] Embedding {base_name} ({len(base_data)} items) ...")

    embedded: List[Dict[str, Any]] = []

    for i, item in enumerate(base_data):
        result_text = item["result_thought"]
        emb = get_embedding(result_text)

        embedded.append(
            {
                "id": item["id"],
                "hop_num": item["hop_num"],
                "hop_text": item["hop_text"],
                "result_thought": result_text,
                "situation": item["cognitive_model"]["5. Situation "],
                "result_embedding": emb,
            }
        )
        if (i + 1) % 50 == 0 or i == len(base_data) - 1:
            print(f"  - Embedded {i+1}/{len(base_data)}")

    save_json(embedded, out_path)
    print(f"[Step2] Saved embedded {base_name} → {out_path}")


def run_step2_embeddings() -> None:
    """
    Run embedding computation for:
        - reasonable_seed_base
        - unreasonable_seed_base
        - retrieve_base
    if not already present.
    """
    print("\n=== Step 2: Compute embeddings for bases ===")

    ensure_exists(REASONABLE_SEED_BASE_PATH, "[Step2]")
    ensure_exists(UNREASONABLE_SEED_BASE_PATH, "[Step2]")
    ensure_exists(RETRIEVE_BASE_PATH, "[Step2]")

    reasonable_seed_base = load_json(REASONABLE_SEED_BASE_PATH)
    unreasonable_seed_base = load_json(UNREASONABLE_SEED_BASE_PATH)
    retrieve_base = load_json(RETRIEVE_BASE_PATH)

    if not os.path.exists(REASONABLE_SEED_EMB_PATH):
        embed_base(reasonable_seed_base, REASONABLE_SEED_EMB_PATH, "reasonable_seed_base")
    else:
        print(f"[Step2] Skipping reasonable_seed_base (already embedded at {REASONABLE_SEED_EMB_PATH})")

    if not os.path.exists(UNREASONABLE_SEED_EMB_PATH):
        embed_base(unreasonable_seed_base, UNREASONABLE_SEED_EMB_PATH, "unreasonable_seed_base")
    else:
        print(f"[Step2] Skipping unreasonable_seed_base (already embedded at {UNREASONABLE_SEED_EMB_PATH})")

    if not os.path.exists(RETRIEVE_BASE_EMB_PATH):
        embed_base(retrieve_base, RETRIEVE_BASE_EMB_PATH, "retrieve_base")
    else:
        print(f"[Step2] Skipping retrieve_base (already embedded at {RETRIEVE_BASE_EMB_PATH})")


# =========================
# Step 3: Retrieve top-k similar examples
# =========================

def build_retrieved_top_k(
    seed_embedded: List[Dict[str, Any]],
    retrieve_embedded: List[Dict[str, Any]],
    out_path: str,
    seed_name: str,
    k: int = TOP_K,
) -> None:
    """
    For each seed embedding, retrieve top-k similar items from retrieve_embedded.
    Save as a list of:
        {
          "seed_id": ...,
          "seed_hop_num": ...,
          "hop_text": ...,
          "top_k_retrieved": ["id-hop_num", ...]
        }
    """
    print(f"\n=== Step 3: Retrieve top-{k} for {seed_name} ===")

    results: List[Dict[str, Any]] = []

    for idx, seed_item in enumerate(seed_embedded):
        seed_emb = seed_item["result_embedding"]

        scores: List[Tuple[float, str, int]] = []
        for ret_item in retrieve_embedded:
            sim = cosine(seed_emb, ret_item["result_embedding"])
            scores.append((sim, ret_item["id"], ret_item["hop_num"]))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        results.append(
            {
                "seed_id": seed_item["id"],
                "seed_hop_num": seed_item["hop_num"],
                "hop_text": seed_item["hop_text"],
                "top_k_retrieved": [f"{s[1]}-{s[2]}" for s in top_k],
            }
        )

        if (idx + 1) % 50 == 0 or idx == len(seed_embedded) - 1:
            print(f"  - Processed {idx+1}/{len(seed_embedded)} seeds")

    save_json(results, out_path)
    print(f"[Step3] Saved retrieved neighbors for {seed_name} → {out_path}")


def run_step3_retrieval() -> None:
    """
    Run retrieval for:
        - reasonable_seed_base_embedded
        - unreasonable_seed_base_embedded
    using retrieve_base_embedded.
    """
    print("\n=== Step 3: Build retrieved neighbors (top-k) ===")

    ensure_exists(REASONABLE_SEED_EMB_PATH, "[Step3]")
    ensure_exists(UNREASONABLE_SEED_EMB_PATH, "[Step3]")
    ensure_exists(RETRIEVE_BASE_EMB_PATH, "[Step3]")

    reasonable_seed_emb = load_json(REASONABLE_SEED_EMB_PATH)
    unreasonable_seed_emb = load_json(UNREASONABLE_SEED_EMB_PATH)
    retrieve_emb = load_json(RETRIEVE_BASE_EMB_PATH)

    if not os.path.exists(REASONABLE_RETRIEVED_TOPK_PATH):
        build_retrieved_top_k(
            reasonable_seed_emb, retrieve_emb,
            REASONABLE_RETRIEVED_TOPK_PATH,
            seed_name="reasonable seeds",
            k=TOP_K,
        )
    else:
        print(f"[Step3] Skipping reasonable retrieval (exists: {REASONABLE_RETRIEVED_TOPK_PATH})")

    if not os.path.exists(UNREASONABLE_RETRIEVED_TOPK_PATH):
        build_retrieved_top_k(
            unreasonable_seed_emb, retrieve_emb,
            UNREASONABLE_RETRIEVED_TOPK_PATH,
            seed_name="unreasonable seeds",
            k=TOP_K,
        )
    else:
        print(f"[Step3] Skipping unreasonable retrieval (exists: {UNREASONABLE_RETRIEVED_TOPK_PATH})")


# =========================
# Step 4: Build seed+retrieved bundles
#         and generate augmented causal thoughts via LLM
# =========================

def build_seed_and_retrieved(
    retrieved_topk: List[Dict[str, Any]],
    seed_base: List[Dict[str, Any]],
    retrieve_base: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each entry in retrieved_topk, attach:
        - full seed_info (from seed_base)
        - list of retrieved_examples (from retrieve_base)

    Output entries have:
        {
          "seed_id",
          "seed_hop_num",
          "seed_hop_text",
          "seed_info",
          "retrieved_examples": [full hop dicts]
        }
    """
    seed_and_retrieved: List[Dict[str, Any]] = []

    # Index bases for quick lookup
    seed_index = {(item["id"], item["hop_num"]): item for item in seed_base}
    retrieve_index = {(item["id"], item["hop_num"]): item for item in retrieve_base}

    for ex in retrieved_topk:
        seed_id = ex["seed_id"]
        seed_hop_num = ex["seed_hop_num"]
        seed_hop_text = ex["hop_text"]

        seed_info = seed_index.get((seed_id, seed_hop_num), None)
        if seed_info is None:
            # Skip if we cannot find the seed (should not happen if data is consistent)
            continue

        retrieved_examples: List[Dict[str, Any]] = []
        for tag in ex["top_k_retrieved"]:
            ex_id, ex_hop_str = tag.split("-")
            ex_hop_num = int(ex_hop_str)
            matched = retrieve_index.get((ex_id, ex_hop_num), None)
            if matched is not None:
                retrieved_examples.append(matched)

        seed_and_retrieved.append(
            {
                "seed_id": seed_id,
                "seed_hop_num": seed_hop_num,
                "seed_hop_text": seed_hop_text,
                "seed_info": seed_info,
                "retrieved_examples": retrieved_examples,
            }
        )

    return seed_and_retrieved


def generate_adapted_causal_thoughts_for_bundle(
    bundle: Dict[str, Any],
) -> List[str]:
    """
    Given one bundle:
        - seed_info (with causal_thought, result_thought, situation)
        - retrieved_examples (list of hops with causal_thought)
    call LLM to generate 5 adapted causal thoughts.

    Returns list of 5 strings (generated thoughts).
    """
    seed_info = bundle["seed_info"]
    seed_situation = seed_info["cognitive_model"]["5. Situation "]

    retrieved_causal_thoughts: List[str] = [
        ex["causal_thought"] for ex in bundle["retrieved_examples"][:TOP_K]
    ]
    # If < TOP_K retrieved, pad by repeating; this should be rare
    if len(retrieved_causal_thoughts) < TOP_K:
        retrieved_causal_thoughts = (retrieved_causal_thoughts * TOP_K)[:TOP_K]

    prompt = f"""You are a reasoning-oriented agent specialized in cognitive modeling.  
Your task is to generate **five distinct, plausible, and contextually adapted causal thoughts** based on a **seed situation** and **five retrieved thoughts**.

## Objective:
You are given five **retrieved causal thoughts** from different contexts.

Your task is to **rewrite each of the five retrieved thoughts** so that they fit the **seed situation**.

## How to proceed:

* Adapt each retrieved causal thought **independently** to match the **seed situation**.  
* Do not alter the structure or psychological tone too much from the original retrieved causal thought.  
* The result should **read as a psychologically natural and situationally coherent thought** that someone might plausibly have **in the given seed situation**.

## Input:

* **Retrieved causal thoughts**:  
  1. {retrieved_causal_thoughts[0]}  
  2. {retrieved_causal_thoughts[1]}  
  3. {retrieved_causal_thoughts[2]}  
  4. {retrieved_causal_thoughts[3]}  
  5. {retrieved_causal_thoughts[4]}  

* **Seed situation**:  
  {seed_situation}

## Output Format:

You must rewrite **each** retrieved causal thought so that it matches the seed situation.  
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

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": prompt}
        ],
    )

    content = response.choices[0].message.content

    # Parse "1. ...\n2. ..." format
    items = re.split(r"\n?\s*\d+\.\s+", content.strip())
    # The first split element is empty or header; skip it
    candidates = [x.strip() for x in items[1:] if x.strip()]

    # If for some reason not exactly 5, we can truncate or pad.
    if len(candidates) >= TOP_K:
        return candidates[:TOP_K]
    elif 0 < len(candidates) < TOP_K:
        # pad by repeating last one
        while len(candidates) < TOP_K:
            candidates.append(candidates[-1])
        return candidates
    else:
        # very unlikely; fallback: use original retrieved causal thoughts
        return retrieved_causal_thoughts


def run_step4_augmentation() -> None:
    """
    Build seed+retrieved bundles for reasonable/unreasonable seeds,
    generate 5 adapted causal thoughts for each bundle,
    and save:
        - reasonable_data_augmentation.json
        - unreasonable_data_augmentation.json
    """
    print("\n=== Step 4: Augment causal thoughts via LLM ===")

    ensure_exists(REASONABLE_RETRIEVED_TOPK_PATH, "[Step4]")
    ensure_exists(UNREASONABLE_RETRIEVED_TOPK_PATH, "[Step4]")
    ensure_exists(REASONABLE_SEED_BASE_PATH, "[Step4]")
    ensure_exists(UNREASONABLE_SEED_BASE_PATH, "[Step4]")
    ensure_exists(RETRIEVE_BASE_PATH, "[Step4]")

    reasonable_retrieved_topk = load_json(REASONABLE_RETRIEVED_TOPK_PATH)
    unreasonable_retrieved_topk = load_json(UNREASONABLE_RETRIEVED_TOPK_PATH)
    reasonable_seed_base = load_json(REASONABLE_SEED_BASE_PATH)
    unreasonable_seed_base = load_json(UNREASONABLE_SEED_BASE_PATH)
    retrieve_base = load_json(RETRIEVE_BASE_PATH)

    # Reasonable
    if not os.path.exists(REASONABLE_AUGMENTED_PATH):
        print("[Step4] Generating augmented thoughts for REASONABLE seeds...")
        reasonable_bundles = build_seed_and_retrieved(
            reasonable_retrieved_topk,
            reasonable_seed_base,
            retrieve_base,
        )

        reasonable_augmented: List[Dict[str, Any]] = []
        for i, bundle in enumerate(reasonable_bundles):
            generated = generate_adapted_causal_thoughts_for_bundle(bundle)

            augmented_format = {
                "seed_id": bundle["seed_id"],
                "seed_hop_num": bundle["seed_hop_num"],
                "seed_causal_thought": bundle["seed_info"]["causal_thought"],
                "seed_result_thought": bundle["seed_info"]["result_thought"],
                "generated_causal_thought": generated,
            }
            reasonable_augmented.append(augmented_format)

            if (i + 1) % 20 == 0 or i == len(reasonable_bundles) - 1:
                print(f"  - Reasonable bundles processed: {i+1}/{len(reasonable_bundles)}")

        save_json(reasonable_augmented, REASONABLE_AUGMENTED_PATH)
        print(f"[Step4] Saved reasonable augmentation → {REASONABLE_AUGMENTED_PATH}")
    else:
        print(f"[Step4] Skipping reasonable augmentation (exists: {REASONABLE_AUGMENTED_PATH})")

    # Unreasonable
    if not os.path.exists(UNREASONABLE_AUGMENTED_PATH):
        print("[Step4] Generating augmented thoughts for UNREASONABLE seeds...")
        unreasonable_bundles = build_seed_and_retrieved(
            unreasonable_retrieved_topk,
            unreasonable_seed_base,
            retrieve_base,
        )

        unreasonable_augmented: List[Dict[str, Any]] = []
        for i, bundle in enumerate(unreasonable_bundles):
            generated = generate_adapted_causal_thoughts_for_bundle(bundle)

            augmented_format = {
                "seed_id": bundle["seed_id"],
                "seed_hop_num": bundle["seed_hop_num"],
                "seed_causal_thought": bundle["seed_info"]["causal_thought"],
                "seed_result_thought": bundle["seed_info"]["result_thought"],
                "generated_causal_thought": generated,
            }
            unreasonable_augmented.append(augmented_format)

            if (i + 1) % 20 == 0 or i == len(unreasonable_bundles) - 1:
                print(f"  - Unreasonable bundles processed: {i+1}/{len(unreasonable_bundles)}")

        save_json(unreasonable_augmented, UNREASONABLE_AUGMENTED_PATH)
        print(f"[Step4] Saved unreasonable augmentation → {UNREASONABLE_AUGMENTED_PATH}")
    else:
        print(f"[Step4] Skipping unreasonable augmentation (exists: {UNREASONABLE_AUGMENTED_PATH})")


# =========================
# Step 5: Build fine-tuning dataset
# =========================

def split_train_valid_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split a dataset into train/valid/test with given ratios.
    """
    random.seed(seed)
    data_copy = data[:]
    random.shuffle(data_copy)

    n = len(data_copy)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    train_set = data_copy[:train_end]
    valid_set = data_copy[train_end:valid_end]
    test_set = data_copy[valid_end:]

    return train_set, valid_set, test_set


def run_step5_build_finetune_dataset() -> None:
    """
    Build instruction fine-tuning data for overpathologizing detection.

    Labeling logic (following the original code):
      - For REASONABLE augmented examples:
          Answer: overpathologized
      - For UNREASONABLE augmented examples:
          Answer: not overpathologized
    """
    print("\n=== Step 5: Build fine-tuning dataset ===")

    ensure_exists(REASONABLE_AUGMENTED_PATH, "[Step5]")
    ensure_exists(UNREASONABLE_AUGMENTED_PATH, "[Step5]")

    reasonable_augmented = load_json(REASONABLE_AUGMENTED_PATH)
    unreasonable_augmented = load_json(UNREASONABLE_AUGMENTED_PATH)

    finetuning_data_reasonable: List[Dict[str, Any]] = []
    for item in reasonable_augmented:
        instruction = f"""You will be given the following:
- A 'target thought' (the resulting thought),
- A set of 5 'plausible causal thoughts' that could reasonably lead to the target thought,
- And 1 additional 'candidate causal thought' whose validity you need to evaluate.

Determine whether the 'candidate causal thought' is reasonably included among the plausible causal thoughts that could give rise to the target thought.
If it's reasonably included, the answer is 'overpathologized'.
If it's not reasonably included, the answer is 'not overpathologized'.

Target Thought (result): {item['seed_result_thought']}
Plausible Causal Thoughts (5): {item['generated_causal_thought']}
Candidate Causal Thought to Evaluate: {item['seed_causal_thought']}
"""
        text = f"[INST] {instruction} [/INST] Answer: overpathologized"
        finetuning_data_reasonable.append({"text": text})

    finetuning_data_unreasonable: List[Dict[str, Any]] = []
    for item in unreasonable_augmented:
        instruction = f"""You will be given the following:
- A 'target thought' (the resulting thought),
- A set of 5 'plausible causal thoughts' that could reasonably lead to the target thought,
- And 1 additional 'candidate causal thought' whose validity you need to evaluate.

Determine whether the 'candidate causal thought' is reasonably included among the plausible causal thoughts that could give rise to the target thought.
If it's reasonably included, the answer is 'overpathologized'.
If it's not reasonably included, the answer is 'not overpathologized'.

Target Thought (result): {item['seed_result_thought']}
Plausible Causal Thoughts (5): {item['generated_causal_thought']}
Candidate Causal Thought to Evaluate: {item['seed_causal_thought']}
"""
        text = f"[INST] {instruction} [/INST] Answer: not overpathologized"
        finetuning_data_unreasonable.append({"text": text})

    total_finetuning_data = finetuning_data_reasonable + finetuning_data_unreasonable
    save_json(total_finetuning_data, FT_ALL_PATH)
    print(f"[Step5] Saved full fine-tuning dataset → {FT_ALL_PATH}")
    print(f"  - overpathologized: {len(finetuning_data_reasonable)}")
    print(f"  - not overpathologized: {len(finetuning_data_unreasonable)}")

    # Split by label (optional) then merge to keep class balance in each split
    reasonable_data = [
        d for d in total_finetuning_data if "Answer: overpathologized" in d["text"]
    ]
    unreasonable_data = [
        d for d in total_finetuning_data if "Answer: not overpathologized" in d["text"]
    ]

    reasonable_train, reasonable_valid, reasonable_test = split_train_valid_test(
        reasonable_data
    )
    unreasonable_train, unreasonable_valid, unreasonable_test = split_train_valid_test(
        unreasonable_data
    )

    train_set = reasonable_train + unreasonable_train
    valid_set = reasonable_valid + unreasonable_valid
    test_set = reasonable_test + unreasonable_test

    random.seed(RANDOM_SEED)
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    print("Train size:", len(train_set))
    print(" - overpathologized:", len(reasonable_train))
    print(" - not overpathologized:", len(unreasonable_train))

    print("Valid size:", len(valid_set))
    print(" - overpathologized:", len(reasonable_valid))
    print(" - not overpathologized:", len(unreasonable_valid))

    print("Test size:", len(test_set))
    print(" - overpathologized:", len(reasonable_test))
    print(" - not overpathologized:", len(unreasonable_test))

    save_json(train_set, FT_TRAIN_PATH)
    save_json(valid_set, FT_VALID_PATH)
    save_json(test_set, FT_TEST_PATH)

    print(f"[Step5] Saved splits:\n"
          f"  train → {FT_TRAIN_PATH}\n"
          f"  valid → {FT_VALID_PATH}\n"
          f"  test  → {FT_TEST_PATH}")


# =========================
# Main
# =========================

def main() -> None:
    ensure_exists(THOUGHT_PATH_DATA_PATH,
                  "[Main] Please generate Thought_Path_Data_ver03.json first.")

    thought_path_data = load_json(THOUGHT_PATH_DATA_PATH)

    # Step 1: split into bases (only if not already done)
    if not (os.path.exists(REASONABLE_SEED_BASE_PATH)
            and os.path.exists(UNREASONABLE_SEED_BASE_PATH)
            and os.path.exists(RETRIEVE_BASE_PATH)):
        split_seed_and_retrieve(thought_path_data)
    else:
        print("[Main] Seed/retrieve bases already exist. Skipping Step 1.")

    # Step 2: embeddings
    run_step2_embeddings()

    # Step 3: retrieval
    run_step3_retrieval()

    # Step 4: augmentation via LLM
    run_step4_augmentation()

    # Step 5: fine-tuning dataset construction
    run_step5_build_finetune_dataset()


if __name__ == "__main__":
    main()