"""
1_Fine_grained_Thought_Path_Generation.py

Inference-time pipeline for:
1) Cognitive Model Extraction (LLM-based)
2) Fine-grained Thought Path Generation (LLM-based)

This script assumes:
- You have the original CDD dataset at: data/Cognitive Distortion Detection.json
- You have an OpenAI API key in the environment: OPENAI_API_KEY
- You will run this script multiple times:
    - First run: creates batch jobs when needed
    - After each batch is finished and its output is downloaded to OUTPUT_DIR,
      set the corresponding batch ID environment variable and re-run.

Batch ID environment variables:
- FINE_COGNITIVE_BATCH_ID   : for Step 1 (cognitive model extraction)
- FINE_THOUGHT_PATH_BATCH_ID: for Step 2 (fine-grained thought path generation)
"""

import os
import json
import re
from copy import deepcopy
from typing import List, Dict, Any

from openai import OpenAI


# =========================
# Paths & configuration
# =========================

DATA_DIR = "data"
OUTPUT_DIR = os.path.join("outputs", "inference_fine_grained")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGINAL_CDD_PATH = os.path.join(DATA_DIR, "Cognitive Distortion Detection.json")

COGNITIVE_MODEL_RAW_PATH = os.path.join(
    OUTPUT_DIR, "fine_cognitive_model_raw.json"
)
COGNITIVE_MODEL_NUMBERED_PATH = os.path.join(
    OUTPUT_DIR, "fine_cognitive_model_numbered.json"
)

THOUGHT_PATH_RAW_PATH = os.path.join(
    OUTPUT_DIR, "fine_thought_path_raw.json"
)
THOUGHT_PATH_NUMBERED_PATH = os.path.join(
    OUTPUT_DIR, "fine_thought_path_numbered.json"
)

# Model names (can be adjusted)
FINE_COGNITIVE_MODEL_NAME = "gpt-4o-mini-2024-07-18"
FINE_THOUGHT_PATH_MODEL_NAME = "gpt-4o-mini-2024-07-18"

DEFAULT_TEMPERATURE = 0.3

# Batch IDs for this script (inference pipeline)
FINE_COGNITIVE_BATCH_ID = os.environ.get("FINE_COGNITIVE_BATCH_ID", None)
FINE_THOUGHT_PATH_BATCH_ID = os.environ.get("FINE_THOUGHT_PATH_BATCH_ID", None)

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


def batch_output_path(batch_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{batch_id}_output.jsonl")


def build_batch_input(
    query_list: List[str],
    model_name: str,
    batch_input_path: str,
    temperature: float | None = None,
) -> None:
    """
    Build a JSONL file for the OpenAI Batch API from a list of prompts.
    """
    init_template = {
        "custom_id": None,  # must be unique within the batch
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [],
        },
    }

    if temperature is not None:
        init_template["body"]["temperature"] = temperature

    batches = []
    for idx, query in enumerate(query_list):
        temp = deepcopy(init_template)
        temp["custom_id"] = f"{idx}"
        temp["body"]["messages"].append({"role": "system", "content": query})
        batches.append(temp)

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for item in batches:
            json_string = json.dumps(item)
            f.write(json_string + "\n")

    print(f"[Batch] Saved input with {len(batches)} items → {batch_input_path}")


def create_batch(batch_input_path: str, description: str) -> str:
    """
    Create a batch job on OpenAI and return the batch id.
    """
    batch_input_file = client.files.create(
        file=open(batch_input_path, "rb"),
        purpose="batch",
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )

    print(f"[Batch] Created batch with id: {batch.id}")
    print("  Please monitor the batch status on the OpenAI Dashboard.")
    print(f"  Once it is completed, download the output as:")
    print(f"    {batch_output_path(batch.id)}")
    print(f"  Then set the corresponding BATCH_ID environment variable to: {batch.id}")
    return batch.id


def load_batch_output_file(batch_id: str) -> List[Dict[str, Any]]:
    """
    Load batch output JSONL into a list of records.
    """
    output_path = batch_output_path(batch_id)
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Batch output file not found: {output_path}\n"
            f"Please download the batch output to this path first."
        )

    batch_data: List[Dict[str, Any]] = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            batch_data.append(json.loads(line.strip()))

    print(f"[Batch] Loaded {len(batch_data)} records from {output_path}")
    return batch_data


# =========================
# Step 1: Fine-grained Cognitive Model Extraction
# =========================

def build_fine_cognitive_model_prompts(
    original_data: List[Dict[str, Any]]
) -> List[str]:
    """
    Build prompts for cognitive model extraction (fine-grained, inference version).
    """
    query_list: List[str] = []

    for item in original_data:
        original = item["Patient Question"]

        prompt_template = f"""You are a CBT-trained language analyst.

Your task is to extract the user's cognitive model from a personal post using only the user's original words or phrasing, without adding or assuming anything that is not explicitly present in the post.

A cognitive model consists of the following 8 components:

1. **Relevant Story**
- Contains significant past events that contribute to an individual's mental state.

2. **Core Beliefs**
- Deeply ingrained perceptions about oneself, others, and the world.
- If no belief is clearly reflected in the post, write **"Can't find"**.

Write the **exact belief**  (e.g., “I am defective”) if found.

3. **Intermediate Beliefs**
- Underlying rules, attitudes, and assumptions that are derived from core beliefs and shape an individual's thought patterns.
- Do not confuse with automatic thoughts.

4. **Coping Strategies**
- Techniques used to manage negative emotions. (e.g., avoidance, suppression).

5. **Situation**
- A specific external event or trigger that occurred.

6. **Automatic Thoughts**
- Quick, evaluative thoughts that are triggered by the situation without conscious deliberation, and that stem from underlying beliefs. 
- **Write only the automatic thoughts of the person who wrote the post.**
- **Completely ignore the speech, thoughts, or experiences of other people mentioned (children, spouse, friends). Do not paraphrase their speech as if it were the author's.**
- Only extract the author's own direct thoughts, worries, assumptions, or interpretations.
- Always write in the first-person perspective (“I ...”).

7. **Emotions**
- The emotional response triggered by the automatic thoughts.
- Select one or more emotional labels from the list below that best match what the user explicitly states or strongly implies.

8. **Behaviors**
- The behavioral responses result from the automatic thoughts.

### General Instructions:
- Never use the words 'user', 'author', or 'their'.
- Always write as if you are the author directly speaking in first-person (“I ...”).
- **If a component contains only another person's experiences and no direct thoughts from the author, write 'Can't find.'**
- Do not invent, assume, or reinterpret.
- Each section must be no more than 5 lines.

Now extract the cognitive model from the following user post:
*User's Post:*
{original}

**Answer format:** (Your answer starts here. Copy the format exactly. Do not add any other text.)

1. Relevant Story: ...
2. Core Beliefs: ...
3. Intermediate Beliefs: ...
4. Coping Strategies: ...
5. Situation: ...
6. Automatic Thoughts: ...
7. Emotions: ...
8. Behaviors: ...
"""
        query_list.append(prompt_template)

    return query_list


def parse_fine_cognitive_model_batch(
    batch_data: List[Dict[str, Any]],
    original_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Parse batch output for fine-grained cognitive model extraction.
    """
    extracted_cognitive_model: List[Dict[str, Any]] = []

    for i, record in enumerate(batch_data):
        text = record["response"]["body"]["choices"][0]["message"]["content"]
        extracted_cognitive_model.append(
            {
                "id": original_data[i]["Id_Number"],
                "cognitive_model": text,
            }
        )

    save_json(extracted_cognitive_model, out_path)
    print(f"[Step1] Saved fine-grained cognitive model → {out_path}")


def split_fine_cognitive_model_sections(
    input_path: str,
    output_path: str,
) -> None:
    """
    Split the textual cognitive model into 8 numbered components based on known prefixes.
    """
    cognitive_model_data = load_json(input_path)

    keys = [
        "1. Relevant Story: ",
        "2. Core Beliefs: ",
        "3. Intermediate Beliefs: ",
        "4. Coping Strategies: ",
        "5. Situation: ",
        "6. Automatic Thoughts: ",
        "7. Emotions: ",
        "8. Behaviors: ",
    ]

    parsed_results: List[Dict[str, Any]] = []

    for item in cognitive_model_data:
        text = item.get("cognitive_model", "")
        id_ = item.get("id", "")

        result: Dict[str, Any] = {"id": id_}

        for i in range(len(keys)):
            start_idx = text.find(keys[i])
            if start_idx == -1:
                result[keys[i].replace(":", "")] = "Not found"
                continue

            start = start_idx + len(keys[i])
            end = text.find(keys[i + 1]) if i + 1 < len(keys) else len(text)
            value = text[start:end].strip()
            key_name = keys[i].replace(":", "")
            result[key_name] = value

        parsed_results.append(result)

    save_json(parsed_results, output_path)
    print(f"[Step1] Saved numbered fine-grained cognitive model → {output_path}")


def run_step1_fine_cognitive_model_extraction(
    original_data: List[Dict[str, Any]]
) -> None:
    """
    Step 1 (inference): Fine-grained Cognitive Model Extraction.

    - If numbered cognitive model already exists: skip.
    - Else if batch output exists (FINE_COGNITIVE_BATCH_ID set & file downloaded): parse it.
    - Else: create a new batch for cognitive model extraction.
    """
    print("\n=== Step 1: Fine-grained Cognitive Model Extraction ===")

    if os.path.exists(COGNITIVE_MODEL_NUMBERED_PATH):
        print(f"[Step1] Found existing numbered fine-grained cognitive model at {COGNITIVE_MODEL_NUMBERED_PATH}. Skipping Step 1.")
        return

    if FINE_COGNITIVE_BATCH_ID is not None:
        try:
            batch_data = load_batch_output_file(FINE_COGNITIVE_BATCH_ID)
            parse_fine_cognitive_model_batch(
                batch_data,
                original_data,
                out_path=COGNITIVE_MODEL_RAW_PATH,
            )
            split_fine_cognitive_model_sections(
                input_path=COGNITIVE_MODEL_RAW_PATH,
                output_path=COGNITIVE_MODEL_NUMBERED_PATH,
            )
            print("[Step1] Completed using existing batch output.")
            return
        except FileNotFoundError as e:
            print(f"[Step1] {e}")

    print("[Step1] No numbered output and no usable batch output found.")
    print("[Step1] Creating a new batch for fine-grained cognitive model extraction...")

    prompts = build_fine_cognitive_model_prompts(original_data)
    batch_input_path = os.path.join(OUTPUT_DIR, "fine_cognitive_model_batch_input.jsonl")
    build_batch_input(
        prompts,
        model_name=FINE_COGNITIVE_MODEL_NAME,
        batch_input_path=batch_input_path,
        temperature=DEFAULT_TEMPERATURE,
    )
    batch_id = create_batch(batch_input_path, "fine cognitive model extraction")

    print("\n[Step1] Please wait for the batch to finish, download the output,")
    print(f"        then set FINE_COGNITIVE_BATCH_ID={batch_id} and re-run the script to complete Step 1.")


# =========================
# Step 2: Fine-grained Thought Path Generation
# =========================

def build_fine_thought_path_prompts(
    parsed_results: List[Dict[str, Any]]
) -> List[str]:
    """
    Build prompts for fine-grained Thought Path generation from the numbered cognitive model.
    """
    query_list: List[str] = []

    for item in parsed_results:
        cognitive_model = ""
        fields = [
            ("1. Relevant Story", "Relevant Story"),
            ("2. Core Beliefs", "Core Beliefs"),
            ("3. Intermediate Beliefs", "Intermediate Beliefs"),
            ("4. Coping Strategies", "Coping Strategies"),
            ("5. Situation", "Situation"),
            ("6. Automatic Thoughts", "Automatic Thoughts"),
            ("7. Emotions", "Emotions"),
            ("8. Behaviors", "Behaviors"),
        ]

        for field_key, _ in fields:
            value = item.get(field_key + " ", "Can't find")
            if value != "Can't find":
                cognitive_model += f"{field_key}: {value} "
            else:
                cognitive_model += f"{field_key}: (Not described) "

        prompt_template = f"""You are a CBT-trained reasoning assistant.

**Task:**
You will be given a user's cognitive model.
Construct a **Thought Path** that shows how the user progressed from the **situation** to one or more of the following components:

1. **Automatic Thoughts** (preferred),
2. **Emotions** (if available),
3. **Behaviors** (if available).

If none of these exist, respond with: **"No distortion"**

**Instructions:**
- Begin from the user's **"5. Situation"**.
- When available, incorporate the user’s **"3. Intermediate Beliefs"** and **"4. Coping Strategies"** as hops in the path:
  - Use Intermediate Beliefs as rule/assumption steps that precede or lead into Automatic Thoughts.
  - Use Coping Strategies only as behavioral strategy steps (e.g., after emotions or alongside behaviors), without adding new strategies.

- Progress through the user's plausible internal steps, marked as hops:

  - situation → ~ → ~ → **automatic thoughts**
  - If **emotions** are available in the cognitive model, continue:
    → **emotions**
  - If **behaviors** are also available, continue:
    → **behaviors**

- Your goal is to reconstruct the likely internal reasoning chain that leads from the **situation** to the user’s **automatic thoughts**, and then optionally to their **emotional** or **behavioral** responses, if present.

**Strict Constraints:**
- **Only use information explicitly found in the cognitive model.**
  Do **not** add, guess, or infer new information (such as feelings, beliefs, or experiences) that are not clearly present in the original cognitive model.

- **Do not speculate.** If the reasoning step or emotional reaction is not directly supported by the cognitive model, leave it out.

- Avoid assumptions or imaginative elaboration — your output must remain grounded strictly in the user’s own reported data.

- If **automatic thoughts** are "(Not described)", then build the path toward **core beliefs**.
  If both are missing, continue toward **intermediate beliefs**.
  If none of these are available, return: **"No distortion"**

- Be careful to preserve the **user's perspective and tone**. Use only plausible psychological transitions — avoid skipping logical or emotional steps.
- Each step should reflect a plausible internal reaction, interpretation, or emotional shift that explains how the user arrived at the final expression.

**Input:**
**User's cognitive model:**
{cognitive_model}

(Your answer starts here. Copy the format exactly.)
**Output Format:**

situation : ...
→ ...
→ ...
→ automatic thoughts : ...
→ (optional) emotions : ...
→ (optional) behaviors : ...
"""
        query_list.append(prompt_template)

    return query_list


def parse_fine_thought_path_batch(
    batch_data: List[Dict[str, Any]],
    original_data: List[Dict[str, Any]],
    out_raw_path: str,
    out_numbered_path: str,
) -> None:
    """
    Parse fine-grained Thought Path batch output:
    - Save raw thought path
    - Produce an alphabet-labeled version with components
    """
    inferenced: List[Dict[str, Any]] = []

    for i, record in enumerate(batch_data):
        inferenced_output = record["response"]["body"]["choices"][0]["message"]["content"]
        inferenced.append(
            {
                "id": original_data[i]["Id_Number"],
                "inferenced": inferenced_output,
            }
        )

    save_json(inferenced, out_raw_path)
    print(f"[Step2] Saved raw fine-grained Thought Path → {out_raw_path}")

    thought_path: List[Dict[str, Any]] = []

    for item in inferenced:
        thought_path_text = item["inferenced"]

        path_dict: Dict[str, Any] = {
            "id": item["id"],
            "fine-grained_thought_path": thought_path_text.strip(),
        }

        raw_lines = [line.strip() for line in thought_path_text.strip().split("→")]

        w_alphabet_lines: List[str] = []
        components: List[str] = []

        for idx, line in enumerate(raw_lines):
            # Remove labels like "situation :" / "automatic thoughts :" if present
            if ":" in line:
                content = line.split(":", 1)[1].strip()
            else:
                content = line.strip()

            components.append(content)

            label = chr(ord("a") + idx)  # (a), (b), ...
            w_alphabet_lines.append(f"({label}) {content}")

        path_dict["w_alphabet"] = "\n→ ".join(w_alphabet_lines)
        path_dict["components"] = components

        thought_path.append(path_dict)

    save_json(thought_path, out_numbered_path)
    print(f"[Step2] Saved numbered fine-grained Thought Path → {out_numbered_path}")


def run_step2_fine_thought_path_generation(
    original_data: List[Dict[str, Any]]
) -> None:
    """
    Step 2 (inference): Fine-grained Thought Path Generation.

    - Requires numbered fine-grained cognitive model from Step 1.
    - If numbered Thought Path already exists: skip.
    - Else if batch output exists (FINE_THOUGHT_PATH_BATCH_ID & file downloaded): parse it.
    - Else: create a new batch for Thought Path generation.
    """
    print("\n=== Step 2: Fine-grained Thought Path Generation ===")

    if not os.path.exists(COGNITIVE_MODEL_NUMBERED_PATH):
        print(f"[Step2] Missing {COGNITIVE_MODEL_NUMBERED_PATH}. Please complete Step 1 first.")
        return

    if os.path.exists(THOUGHT_PATH_NUMBERED_PATH):
        print(f"[Step2] Found existing fine-grained Thought Path at {THOUGHT_PATH_NUMBERED_PATH}. Skipping Step 2.")
        return

    parsed_results = load_json(COGNITIVE_MODEL_NUMBERED_PATH)

    if FINE_THOUGHT_PATH_BATCH_ID is not None:
        try:
            batch_data = load_batch_output_file(FINE_THOUGHT_PATH_BATCH_ID)
            parse_fine_thought_path_batch(
                batch_data,
                original_data,
                out_raw_path=THOUGHT_PATH_RAW_PATH,
                out_numbered_path=THOUGHT_PATH_NUMBERED_PATH,
            )
            print("[Step2] Completed using existing batch output.")
            return
        except FileNotFoundError as e:
            print(f"[Step2] {e}")

    print("[Step2] No Thought Path output and no usable batch output found.")
    print("[Step2] Creating a new batch for fine-grained Thought Path generation...")

    prompts = build_fine_thought_path_prompts(parsed_results)
    batch_input_path = os.path.join(OUTPUT_DIR, "fine_thought_path_batch_input.jsonl")
    build_batch_input(
        prompts,
        model_name=FINE_THOUGHT_PATH_MODEL_NAME,
        batch_input_path=batch_input_path,
        temperature=DEFAULT_TEMPERATURE,
    )
    batch_id = create_batch(batch_input_path, "fine thought path generation")

    print("\n[Step2] Please wait for the batch to finish, download the output,")
    print(f"        then set FINE_THOUGHT_PATH_BATCH_ID={batch_id} and re-run the script to complete Step 2.")


# =========================
# Main
# =========================

def main() -> None:
    if not os.path.exists(ORIGINAL_CDD_PATH):
        raise FileNotFoundError(
            f"Original CDD file not found: {ORIGINAL_CDD_PATH}\n"
            f"Please place 'Cognitive Distortion Detection.json' under the 'data/' directory."
        )

    original_data = load_json(ORIGINAL_CDD_PATH)

    # Step 1: fine-grained cognitive model
    run_step1_fine_cognitive_model_extraction(original_data)

    # Step 2: fine-grained Thought Path generation
    run_step2_fine_thought_path_generation(original_data)


if __name__ == "__main__":
    main()
