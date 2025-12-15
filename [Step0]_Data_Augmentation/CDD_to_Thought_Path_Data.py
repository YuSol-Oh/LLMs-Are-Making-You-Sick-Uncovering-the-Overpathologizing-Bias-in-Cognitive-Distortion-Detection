"""
CDD_to_Thought_Path_Data.py

This script converts the original Cognitive Distortion Detection (CDD) dataset into
Thought Path Data via the following steps:

1) Cognitive Model Extraction (LLM-based)
2) Fine-grained Thought Path Generation (LLM-based)
3) Hop-level Labeling (LLM-based)
4) Constructing the final Thought Path dataset

Usage overview (high-level):

- Put the original dataset at: data/Cognitive Distortion Detection.json
- Set OPENAI_API_KEY in your environment.
- Optionally set:
    COGNITIVE_BATCH_ID
    THOUGHT_PATH_BATCH_ID
    HOP_LABEL_BATCH_ID
  after each batch is completed and its output file is downloaded into OUTPUT_DIR.

The script is designed so that:
- All steps can be "on" in main().
- Each step checks its own preconditions:
  - Required input files exist
  - Batch output file exists (based on batch ID)
  - If not available, a new batch is created and the user is instructed what to do next.
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
OUTPUT_DIR = os.path.join("outputs", "step0_data_augmentation", "ver03")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGINAL_CDD_PATH = os.path.join(DATA_DIR, "Cognitive Distortion Detection.json")

COGNITIVE_MODEL_RAW_PATH = os.path.join(
    OUTPUT_DIR, "extracted_cognitive_model.json"
)
COGNITIVE_MODEL_NUMBERED_PATH = os.path.join(
    OUTPUT_DIR, "extracted_cognitive_model_numbered.json"
)

THOUGHT_PATH_RAW_PATH = os.path.join(
    OUTPUT_DIR, "generated_Thought_Path.json"
)
THOUGHT_PATH_NUMBERED_PATH = os.path.join(
    OUTPUT_DIR, "generated_Thought_Path_numbered.json"
)

LABELED_HOP_PATH = os.path.join(
    OUTPUT_DIR, "labeled_hop.json"
)

FINAL_THOUGHT_PATH_DATA_PATH = os.path.join(
    OUTPUT_DIR, "Thought_Path_Data_ver03.json"
)

# Model names (can be adjusted)
COGNITIVE_MODEL_MODEL_NAME = "gpt-4o-mini-2024-07-18"
THOUGHT_PATH_MODEL_NAME = "gpt-4o-mini-2024-07-18"
HOP_LABEL_MODEL_NAME = "gpt-4o-mini-2024-07-18"

DEFAULT_TEMPERATURE = 0.3

# Batch IDs are read from environment variables (or can be set here manually)
COGNITIVE_BATCH_ID = os.environ.get("COGNITIVE_BATCH_ID", None)
THOUGHT_PATH_BATCH_ID = os.environ.get("THOUGHT_PATH_BATCH_ID", None)
HOP_LABEL_BATCH_ID = os.environ.get("HOP_LABEL_BATCH_ID", None)

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

    batch_data = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            batch_data.append(json.loads(line.strip()))

    print(f"[Batch] Loaded {len(batch_data)} records from {output_path}")
    return batch_data


# =========================
# Cognitive Model Extraction (Step 1)
# =========================

def build_cognitive_model_prompts(original_data: List[Dict[str, Any]]) -> List[str]:
    """
    Build prompts for cognitive model extraction for each example in the original CDD data.
    """
    query_list: List[str] = []

    for item in original_data:
        original = item["Patient Question"]
        distorted_part = item["Distorted part"]

        prompt_template = f"""You are a CBT-trained language analyst.

Your task is to extract the user's cognitive model from a personal post using only the user's original words or phrasing, without adding or assuming anything that is not explicitly present in the post.

### Step 1: Identify the User
- First, explicitly identify who the author of the post (the “user”) is.  
- The user is always the person writing in the first-person perspective (“I”).  
- Other people mentioned in the post (e.g., children, spouse, friends) are **not the user**.  
- All subsequent components must be extracted **strictly from the user’s perspective only**.  

### Step 2: Extract the Cognitive Model
A cognitive model consists of the following 8 components:

1. **Relevant Story**  
   - Contains significant past events that contribute to an individual's mental state.

2. **Core Beliefs**  
   - Deeply ingrained perceptions about oneself, others, and the world.
   - Select the one that best matches from the list below.
   - If no belief is clearly reflected in the post, write **"Can't find"**.

Core Belief Categories (3 major → 19 subtypes):

**[Helpless]**  
- I am incompetent.  
- I am helpless.  
- I am powerless / weak / vulnerable.  
- I am a victim.  
- I am needy.  
- I am trapped.  
- I am out of control.  
- I am a failure / loser.  
- I am defective.

**[Unlovable]**  
- I am unlovable.  
- I am unattractive.  
- I am undesirable / unwanted.  
- I am bound to be rejected.  
- I am bound to be abandoned.  
- I am bound to be alone.

**[Worthless]**  
- I am worthless / waste.  
- I am immoral.  
- I am bad / dangerous / toxic / evil.  
- I don’t deserve to live.

Write the **exact belief** (e.g., “I am defective”) if found.

3. **Intermediate Beliefs**  
   - Underlying rules, attitudes, and assumptions that are derived from core beliefs and shape an individual's thought patterns.
   - Do not confuse with automatic thoughts.

4. **Coping Strategies**  
   - Techniques used to manage negative emotions. (e.g., avoidance, suppression).

5. **Situation**  
   - A specific external event or trigger that occurred.

6. **Automatic Thoughts**  
   - Quick, evaluative thoughts that are triggered by the situation without conscious deliberation, and that stem from underlying beliefs.  
   - **You must extract only the automatic thoughts of the user who wrote the post.**  
   - Automatic Thoughts must always be from the **first-person perspective of the post’s author (the user)**.  
   - Do **not** include the speech, thoughts, or experiences of other people (e.g., children, spouse, friends).  
   - If the user quotes another person, ignore those quotes unless the user explicitly reacts to them with their own thought, interpretation, or concern.  

7. **Emotions**  
   - The emotional response triggered by the automatic thoughts.
   - Select one or more emotional labels from the list below that best match what the user explicitly states or strongly implies.
   - Use only words from this fixed set. If no emotion is clearly expressed or implied, write **"Can't find"**.

8. **Behaviors**  
   - The behavioral responses result from the automatic thoughts.

### Additional Instruction:

**Distorted Part Inclusion Rule**:
You will also be given a **“distorted part”**, which highlights a potentially distorted portion of the user’s thinking.
You must ensure that **this distorted part is explicitly included in at least one of the following sections**:

- **6. Automatic Thoughts**
- **7. Emotions**
- **8. Behaviors**

Do **not omit or reinterpret** the distorted part. It must appear exactly or very closely in one or more of those sections.

### General Instructions:

- **Automatic Thoughts must always represent the thinking of the post’s author only, never other characters.**  
- Use **only the user’s exact words** or very minimal paraphrasing that preserves the original meaning.
- If no information is found for a component, write **"Can't find"**.
- Never invent, assume, or reinterpret the user's intent.
- Each section must be **no more than 5 lines**.
- Write from the user’s **first-person perspective** throughout.

Now extract the cognitive model from the following user post:

**User’s Post:**
{original}

**Distorted Part (must appear in 6, 7, or 8):**
{distorted_part}

**Answer format:** (Your answer starts here. Copy the format exactly. Do not add any other text.)

Step 1 (User Identification): ...

Step 2 (Extracted Cognitive Model)
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


def parse_cognitive_model_batch(batch_data: List[Dict[str, Any]],
                                original_data: List[Dict[str, Any]],
                                out_path: str) -> None:
    """
    Parse batch output for cognitive model extraction and align with original data by index.
    """
    extracted_cognitive_model = []

    for i, record in enumerate(batch_data):
        text = record["response"]["body"]["choices"][0]["message"]["content"]
        # Split off the second step
        cognitive_model = text.split("Step 2 (Extracted Cognitive Model)\n", 1)[1]
        extracted_cognitive_model.append(
            {
                "id": original_data[i]["Id_Number"],
                "cognitive_model": cognitive_model,
            }
        )

    save_json(extracted_cognitive_model, out_path)
    print(f"[Step1] Saved raw cognitive model → {out_path}")


def split_cognitive_model_sections(input_path: str, output_path: str) -> None:
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

    parsed_results = []

    for item in cognitive_model_data:
        text = item.get("cognitive_model", "")
        id_ = item.get("id", "")

        result = {"id": id_}

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
    print(f"[Step1] Saved numbered cognitive model → {output_path}")


def run_step1_cognitive_model_extraction(original_data: List[Dict[str, Any]]) -> None:
    """
    Step 1: Cognitive Model Extraction (with precondition checks).

    - If output (numbered cognitive model) already exists: skip.
    - Else if batch output exists (batch_id given & file downloaded): parse it.
    - Else: create a new batch input and submit a batch job.
    """
    print("\n=== Step 1: Cognitive Model Extraction ===")

    if os.path.exists(COGNITIVE_MODEL_NUMBERED_PATH):
        print(f"[Step1] Found existing numbered cognitive model at {COGNITIVE_MODEL_NUMBERED_PATH}. Skipping Step 1.")
        return

    # If batch output is ready, parse it
    if COGNITIVE_BATCH_ID is not None:
        try:
            batch_data = load_batch_output_file(COGNITIVE_BATCH_ID)
            parse_cognitive_model_batch(batch_data, original_data, COGNITIVE_MODEL_RAW_PATH)
            split_cognitive_model_sections(COGNITIVE_MODEL_RAW_PATH, COGNITIVE_MODEL_NUMBERED_PATH)
            print("[Step1] Completed using existing batch output.")
            return
        except FileNotFoundError as e:
            print(f"[Step1] {e}")

    # Otherwise, create a new batch
    print("[Step1] No numbered output and no usable batch output found.")
    print("[Step1] Creating a new batch for cognitive model extraction...")

    prompts = build_cognitive_model_prompts(original_data)
    batch_input_path = os.path.join(OUTPUT_DIR, "cognitive_model_batch_input.jsonl")
    build_batch_input(
        prompts,
        model_name=COGNITIVE_MODEL_MODEL_NAME,
        batch_input_path=batch_input_path,
        temperature=DEFAULT_TEMPERATURE,
    )
    batch_id = create_batch(batch_input_path, "cognitive model extraction")

    print("\n[Step1] Please wait for the batch to finish, download the output,")
    print(f"        then set COGNITIVE_BATCH_ID={batch_id} and re-run the script to complete Step 1.")


# =========================
# Thought Path Generation (Step 2)
# =========================

def build_thought_path_prompts(cognitive_model_data: List[Dict[str, Any]]) -> List[str]:
    """
    Build prompts for fine-grained Thought Path generation from the cognitive model.
    """
    query_list: List[str] = []

    for item in cognitive_model_data:
        cognitive_model_str = ""
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
                cognitive_model_str += f"{field_key}: {value} "
            else:
                cognitive_model_str += f"{field_key}: (Not described) "

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

- Progress through the user's plausible internal steps, marked as hops:

  - situation → ~ → ~ → **automatic thoughts**
  - If **emotions** are available in the cognitive model, continue:
    → **emotions**
  - If **behaviors** are also available, continue:
    → **behaviors**

- Your goal is to reconstruct the likely internal reasoning chain that leads from the **situation** to the user’s **automatic thoughts**, and then optionally to their **emotional** or **behavioral** responses, if present.

- If **automatic thoughts** are not available, then build the path toward **core beliefs**.
  If both are missing, continue toward **intermediate beliefs**.
  If none of these are available, return: **"No distortion"**

- Be careful to preserve the **user's perspective and tone**. Use only plausible psychological transitions — avoid skipping logical or emotional steps.
- Each step should reflect a plausible internal reaction, interpretation, or emotional shift that explains how the user arrived at the final expression.

**Input:**
**User's cognitive model:**
{cognitive_model_str}

**Output Format:**

situation : ...
→ ...
→ ...
→ automatic thoughts : ...
→ (optional) emotions : ...
→ (optional) behaviors : ...

If automatic thoughts are missing, follow this format instead:

situation : ...
→ ...
→ ...
→ core beliefs or intermediate beliefs : ...

If none of the above are available, return:
**"No distortion"**
"""
        query_list.append(prompt_template)

    return query_list


def remove_prefix_label(text: str) -> str:
    lower = text.lower().strip()
    prefixes = [
        "situation",
        "automatic thoughts",
        "emotions",
        "emotion",
        "behaviors",
        "behavior",
    ]
    for p in prefixes:
        if lower.startswith(p + " :") or lower.startswith(p + ":"):
            return text.split(":", 1)[-1].strip()
    return text.strip()


def parse_thought_path_batch(batch_data: List[Dict[str, Any]],
                             original_data: List[Dict[str, Any]],
                             out_raw_path: str,
                             out_numbered_path: str) -> None:
    """
    Parse Thought Path batch output and generate:
    - raw Thought Path text
    - numbered version with hop labels (a, b, c, ...)
    - list of components per hop
    """
    inferenced = []
    for i, record in enumerate(batch_data):
        inferenced_output = record["response"]["body"]["choices"][0]["message"]["content"]
        inferenced.append(
            {
                "id": original_data[i]["Id_Number"],
                "inferenced": inferenced_output,
            }
        )

    save_json(inferenced, out_raw_path)
    print(f"[Step2] Saved raw Thought Path → {out_raw_path}")

    thought_path = []
    for item in inferenced:
        thought_path_text = item["inferenced"]

        path_dict = {
            "id": item["id"],
            "fine-grained_thought_path": thought_path_text.strip(),
        }

        raw_lines = [line.strip() for line in thought_path_text.strip().split("→")]

        w_alphabet_lines = []
        components = []

        for idx, line in enumerate(raw_lines):
            if idx == 0:
                w_alphabet_lines.append(line)
                components.append(remove_prefix_label(line))
            elif idx == len(raw_lines) - 1:
                w_alphabet_lines.append(f"→ {line}")
                components.append(remove_prefix_label(line))
            else:
                alphabet = chr(ord("a") + idx - 1)
                w_alphabet_lines.append(f"→ ({alphabet}) {line}")
                components.append(remove_prefix_label(line))

        path_dict["w_alphabet"] = "\n".join(w_alphabet_lines)
        path_dict["components"] = components

        thought_path.append(path_dict)

    save_json(thought_path, out_numbered_path)
    print(f"[Step2] Saved numbered Thought Path → {out_numbered_path}")


def run_step2_thought_path_generation(original_data: List[Dict[str, Any]]) -> None:
    """
    Step 2: Thought Path Generation with precondition checks.

    - Requires the numbered cognitive model JSON from Step 1.
    - If numbered Thought Path already exists: skip.
    - Else if batch output exists: parse it.
    - Else: create a new batch for Thought Path generation.
    """
    print("\n=== Step 2: Thought Path Generation ===")

    if not os.path.exists(COGNITIVE_MODEL_NUMBERED_PATH):
        print(f"[Step2] Missing {COGNITIVE_MODEL_NUMBERED_PATH}. Please complete Step 1 first.")
        return

    if os.path.exists(THOUGHT_PATH_NUMBERED_PATH):
        print(f"[Step2] Found existing Thought Path at {THOUGHT_PATH_NUMBERED_PATH}. Skipping Step 2.")
        return

    cognitive_model_data = load_json(COGNITIVE_MODEL_NUMBERED_PATH)

    # If batch output exists, parse it
    if THOUGHT_PATH_BATCH_ID is not None:
        try:
            batch_data = load_batch_output_file(THOUGHT_PATH_BATCH_ID)
            parse_thought_path_batch(
                batch_data,
                original_data,
                out_raw_path=THOUGHT_PATH_RAW_PATH,
                out_numbered_path=THOUGHT_PATH_NUMBERED_PATH,
            )
            print("[Step2] Completed using existing batch output.")
            return
        except FileNotFoundError as e:
            print(f"[Step2] {e}")

    # Otherwise, create a new batch
    print("[Step2] No Thought Path output and no usable batch output found.")
    print("[Step2] Creating a new batch for Thought Path generation...")

    prompts = build_thought_path_prompts(cognitive_model_data)
    batch_input_path = os.path.join(OUTPUT_DIR, "thought_path_batch_input.jsonl")
    build_batch_input(
        prompts,
        model_name=THOUGHT_PATH_MODEL_NAME,
        batch_input_path=batch_input_path,
    )
    batch_id = create_batch(batch_input_path, "thought path generation")

    print("\n[Step2] Please wait for the batch to finish, download the output,")
    print(f"        then set THOUGHT_PATH_BATCH_ID={batch_id} and re-run the script to complete Step 2.")


# =========================
# Hop-level Labeling (Step 3)
# =========================

def build_hop_label_prompts(original_data: List[Dict[str, Any]],
                            thought_path: List[Dict[str, Any]]) -> List[str]:
    """
    Build prompts for hop-level labeling using the Distorted part and distortion labels.
    """
    query_list: List[str] = []

    for idx, item in enumerate(original_data):
        thought_components = thought_path[idx]["components"]

        hop_list: List[str] = []

        # Remove trailing components such as "Can't find." or "No distortion"
        stop_tokens = ["Can't find.", "No distortion", "Can't find"]

        last_idx = len(thought_components)
        for x in reversed(range(len(thought_components))):
            if thought_components[x].strip() in stop_tokens:
                last_idx = x
            else:
                break

        for x in range(last_idx - 1):
            hop = f"{thought_components[x]} → {thought_components[x + 1]}"
            hop_list.append(hop)

        distorted_part = item["Distorted part"]
        distortion_label = (
            item["Dominant Distortion"]
            + ", "
            + item["Secondary Distortion (Optional)"]
        )

        for hop in hop_list:
            prompt_template = f"""You are a CBT therapist specializing in cognitive distortion detection.

## Task:
You will be given:
- A single **thought hop** (i.e., a transition from one thought to the next)
- A known **distorted part** from the user's original narrative
- The associated **distortion label(s)** (can be one or two)

Your job is to determine whether this hop reflects the **same distortion** as the distorted part.

## Matching Criteria:
- If the **hop text is exactly the same as** or **semantically overlaps** with the distorted part, label it with the given distortion label.
- The overlap must be clear and specific — **general context similarity is NOT enough.**
- You may assign **one** or **both** distortion types to the hop, depending on whether the hop reflects one or both distortions.
- If the hop does **not** contain or paraphrase the distorted part meaningfully, label it as **"No distortion"**

## Input:
- Hop: {hop}
- Distorted part: {distorted_part}
- Distortion type(s): {distortion_label}

## Output format:
1) [Distortion Type(s) or No distortion]  
2) [Short reason why you labeled it this way]

## Output:
(Your answer starts here. Follow the format exactly.)
"""
            query_list.append(prompt_template)

    return query_list


def parse_hop_label_batch(batch_data: List[Dict[str, Any]],
                          original_data: List[Dict[str, Any]],
                          thought_path: List[Dict[str, Any]],
                          out_path: str) -> None:
    """
    Parse hop-level labeling batch output.
    Assumes the batch order is:
      all hops from original_data[0], then original_data[1], ...
    """
    labeled = []
    idx = 0  # index into batch_data

    for z, item in enumerate(original_data):
        id_val = item["Id_Number"]

        thought_components = thought_path[z]["components"]
        stop_tokens = ["Can't find.", "No distortion", "Can't find"]

        last_idx = len(thought_components)
        for x in reversed(range(len(thought_components))):
            if thought_components[x].strip() in stop_tokens:
                last_idx = x
            else:
                break

        hop_count = last_idx - 1
        for hop_num in range(hop_count):
            inferenced_output = batch_data[idx]["response"]["body"]["choices"][0]["message"]["content"]
            labeled.append(
                {
                    "id": id_val,
                    "hop_num": hop_num + 1,  # 1-based index
                    "inferenced": inferenced_output,
                }
            )
            idx += 1

    save_json(labeled, out_path)
    print(f"[Step3] Saved labeled hops → {out_path}")


def run_step3_hop_labeling(original_data: List[Dict[str, Any]]) -> None:
    """
    Step 3: Hop-level Labeling with precondition checks.

    - Requires the numbered Thought Path from Step 2.
    - If labeled hops JSON already exists: skip.
    - Else if batch output exists: parse it.
    - Else: create a new batch.
    """
    print("\n=== Step 3: Hop-level Labeling ===")

    if not os.path.exists(THOUGHT_PATH_NUMBERED_PATH):
        print(f"[Step3] Missing {THOUGHT_PATH_NUMBERED_PATH}. Please complete Step 2 first.")
        return

    if os.path.exists(LABELED_HOP_PATH):
        print(f"[Step3] Found existing labeled hops at {LABELED_HOP_PATH}. Skipping Step 3.")
        return

    thought_path = load_json(THOUGHT_PATH_NUMBERED_PATH)

    # If batch output exists, parse it
    if HOP_LABEL_BATCH_ID is not None:
        try:
            batch_data = load_batch_output_file(HOP_LABEL_BATCH_ID)
            parse_hop_label_batch(batch_data, original_data, thought_path, LABELED_HOP_PATH)
            print("[Step3] Completed using existing batch output.")
            return
        except FileNotFoundError as e:
            print(f"[Step3] {e}")

    # Otherwise, create a new batch
    print("[Step3] No labeled hop output and no usable batch output found.")
    print("[Step3] Creating a new batch for hop-level labeling...")

    prompts = build_hop_label_prompts(original_data, thought_path)
    batch_input_path = os.path.join(OUTPUT_DIR, "hop_label_batch_input.jsonl")
    build_batch_input(
        prompts,
        model_name=HOP_LABEL_MODEL_NAME,
        batch_input_path=batch_input_path,
        temperature=DEFAULT_TEMPERATURE,
    )
    batch_id = create_batch(batch_input_path, "hop-level labeling")

    print("\n[Step3] Please wait for the batch to finish, download the output,")
    print(f"        then set HOP_LABEL_BATCH_ID={batch_id} and re-run the script to complete Step 3.")


# =========================
# Final Thought Path Dataset (Step 4)
# =========================

def build_final_thought_path_dataset(
    labeled_hop_path: str = LABELED_HOP_PATH,
    thought_path_numbered_path: str = THOUGHT_PATH_NUMBERED_PATH,
    cognitive_model_numbered_path: str = COGNITIVE_MODEL_NUMBERED_PATH,
    out_path: str = FINAL_THOUGHT_PATH_DATA_PATH,
) -> None:
    """
    Construct the final Thought Path dataset with hop-level labels and cognitive model context.
    """
    print("\n=== Step 4: Build Final Thought Path Dataset ===")

    if not os.path.exists(labeled_hop_path):
        print(f"[Step4] Missing {labeled_hop_path}. Please complete Step 3 first.")
        return
    if not os.path.exists(thought_path_numbered_path):
        print(f"[Step4] Missing {thought_path_numbered_path}. Please complete Step 2 first.")
        return
    if not os.path.exists(cognitive_model_numbered_path):
        print(f"[Step4] Missing {cognitive_model_numbered_path}. Please complete Step 1 first.")
        return
    if os.path.exists(out_path):
        print(f"[Step4] Final Thought Path dataset already exists at {out_path}. Skipping Step 4.")
        return

    labeled_hop_data = load_json(labeled_hop_path)
    thought_path = load_json(thought_path_numbered_path)
    cognitive_model_data = load_json(cognitive_model_numbered_path)

    hop_text_data: List[str] = []

    # Build hop text list in the same order as in hop_label prompts
    for k in range(len(thought_path)):
        thought_components = thought_path[k]["components"]
        stop_tokens = ["Can't find.", "No distortion", "Can't find"]

        last_idx = len(thought_components)
        for l in reversed(range(len(thought_components))):
            if thought_components[l].strip() in stop_tokens:
                last_idx = l
            else:
                break

        for l in range(last_idx - 1):
            hop = f"{thought_components[l]} → {thought_components[l + 1]}"
            hop_text_data.append(hop)

    final_thought_path_data: List[Dict[str, Any]] = []
    hop_text_index = 0

    # Index cognitive models by id for quick lookup
    cog_model_by_id = {entry["id"]: entry for entry in cognitive_model_data}

    for hop_data in labeled_hop_data:
        hop_id = hop_data["id"]
        hop_num = hop_data["hop_num"]

        # Parse distortion label from the LLM output
        labeled_inference = hop_data["inferenced"]
        pattern = r"(1\)|2\))"
        parts = re.split(pattern, labeled_inference)
        result = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        hop_label = (result.get("1)", "No distortion") or "").strip()

        # Assign hop text
        hop_text = hop_text_data[hop_text_index] if hop_text_index < len(hop_text_data) else "MISSING"
        hop_text_index += 1

        # Split into causal and result thoughts
        causal_thought, result_thought = [part.strip() for part in hop_text.split("→", 1)]

        # Attach cognitive model information
        cog = cog_model_by_id.get(hop_id, {})
        cognitive_model = {
            "5. Situation ": cog.get("5. Situation ", ""),
            "6. Automatic Thoughts ": cog.get("6. Automatic Thoughts ", ""),
            "7. Emotions ": cog.get("7. Emotions ", ""),
        }

        if hop_label.lower() == "no distortion":
            final_thought_path_data.append(
                {
                    "id": hop_id,
                    "hop_num": hop_num,
                    "hop_text": hop_text,
                    "distortion": "No Distortion",
                    "causal_thought": causal_thought,
                    "result_thought": result_thought,
                    "cognitive_model": cognitive_model,
                }
            )
        else:
            # Multiple labels separated by commas
            labels = [lbl.strip() for lbl in hop_label.split(",") if lbl.strip()]
            final_thought_path_data.append(
                {
                    "id": hop_id,
                    "hop_num": hop_num,
                    "hop_text": hop_text,
                    "distortion": "Yes Distortion",
                    "causal_thought": causal_thought,
                    "result_thought": result_thought,
                    "distortion_label": labels,
                    "cognitive_model": cognitive_model,
                }
            )

    save_json(final_thought_path_data, out_path)
    print(f"[Step4] Saved final Thought Path dataset → {out_path}")


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

    # Run all steps; each step checks its own preconditions.
    run_step1_cognitive_model_extraction(original_data)
    run_step2_thought_path_generation(original_data)
    run_step3_hop_labeling(original_data)
    build_final_thought_path_dataset()


if __name__ == "__main__":
    main()
