import os
import time
import torch
import random
import pandas as pd
import json
from tqdm import tqdm
import outlines
from outlines.models import transformers
from transformers import BitsAndBytesConfig, AutoProcessor, MllamaForConditionalGeneration
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gender", choices=["male", "female", "both"], default="both", help="Which gender descriptor pairs to run.")
args = parser.parse_args()


# ------------------------------------------------------------------------------
# (A) Set cache directory
# ------------------------------------------------------------------------------
os.environ["OUTLINES_CACHE_DIR"] = f"/tmp/.outlines_{os.getpid()}"
os.makedirs(os.environ["OUTLINES_CACHE_DIR"], exist_ok=True)

# ------------------------------------------------------------------------------
# (B) Initialize Model
# ------------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
CACHE_DIR = "/scratch/craj/model_cache/llama3"
os.makedirs(CACHE_DIR, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

model = transformers(
    MODEL_NAME,
    model_class=MllamaForConditionalGeneration,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": bnb_config,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "cache_dir": CACHE_DIR,
    },
    # processor_kwargs={"cache_dir": CACHE_DIR},
    device="cuda",
)

SYSTEM_PROMPT = "You are a helpful assistant. Follow the instructions and help the user with the task."
axis = "nationality"
BATCH_SIZE = 1

# ------------------------------------------------------------------------------
# (C) Load descriptors
# ------------------------------------------------------------------------------
descriptor_df = pd.read_csv("../../data/descriptors/use_these_to_gen_images/text_descriptors.csv")

# print("Available columns:", descriptor_df.columns.tolist())
# exit()

descriptor_df.columns = descriptor_df.columns.str.strip()

# Generate same-gender, cross-nationality pairs
man_descriptors = [f"{row['nationality']} {row['man_suffix']}".strip() for _, row in descriptor_df.iterrows()]
woman_descriptors = [f"{row['nationality']} {row['woman_suffix']}".strip() for _, row in descriptor_df.iterrows()]

# Create gender-specific pairs
male_pairs = []
female_pairs = []

for i in range(len(man_descriptors)):
    for j in range(i + 1, len(man_descriptors)):
        male_pairs.append((man_descriptors[i], man_descriptors[j]))

for i in range(len(woman_descriptors)):
    for j in range(i + 1, len(woman_descriptors)):
        female_pairs.append((woman_descriptors[i], woman_descriptors[j]))



# ------------------------------------------------------------------------------
# (D) Load question templates
# ------------------------------------------------------------------------------
with open("../../data/activities/roles.json", "r") as f:
    roles_data = json.load(f)

question_keys = [
    "ability", "ability_irr1", "ability_irr2", "ability_irr3",
    "sociability", "morality", "agency", "status", "appearance"
]

category_mapping = {
    "Necessary Time": ["Contracted Time", "Committed Time", "Free Time"],
    "Contracted Time": ["Necessary Time", "Committed Time", "Free Time"],
    "Committed Time": ["Necessary Time", "Contracted Time", "Free Time"],
    "Free Time": ["Necessary Time", "Contracted Time", "Committed Time"]
}

# ------------------------------------------------------------------------------
# (E) Helper functions
# ------------------------------------------------------------------------------
def get_random_role(category):
    if category in roles_data and roles_data[category]:
        return random.choice(roles_data[category])["role"]
    return None

@torch.inference_mode()
def ask_question_single(messages, pattern):
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    description_generator = outlines.generate.regex(
        model,
        pattern,
        sampler=outlines.samplers.MultinomialSampler(temperature=0.7),
    )
    responses = description_generator([text_prompt])
    return responses[0]

# ------------------------------------------------------------------------------
# (F) Output setup
# ------------------------------------------------------------------------------
output_dir = f"../../outputs/decisionmaking/llama/{axis}_textonly"
os.makedirs(output_dir, exist_ok=True)

# Output setup per gender
processed_sets = {}

for gender in ["male", "female"]:
    progress_log_path = os.path.join(output_dir, f"{axis}_textonly_progress_log_{gender}.csv")
    if os.path.exists(progress_log_path):
        processed_sets[gender] = set(pd.read_csv(progress_log_path)["pair_id"].dropna().tolist())
    else:
        processed_sets[gender] = set()


# ------------------------------------------------------------------------------
# (G) Main loop
# ------------------------------------------------------------------------------
# for gender, gender_pairs in [("male", male_pairs), ("female", female_pairs)]:
gender_map = {
    "male": male_pairs,
    "female": female_pairs
}

selected_genders = ["male", "female"] if args.gender == "both" else [args.gender]

for gender in selected_genders:
    gender_pairs = gender_map[gender]
    results = []
    pair_id = 0
    output_csv_path = os.path.join(output_dir, f"{axis}_textonly_1a_llava_responses_{gender}.csv")
    progress_log_path = os.path.join(output_dir, f"{axis}_textonly_progress_log_{gender}.csv")
    for descriptor_index, (descriptor1, descriptor2) in enumerate(tqdm(gender_pairs[:])):
        pair_id += 1
        pair_key = f"{gender}_{pair_id}"
        if pair_key in processed_sets[gender]:
            continue

        for category, role_list in roles_data.items():
            for role_data in role_list:
                role = role_data["role"]
                for key in question_keys:
                    if key.startswith("ability_irr"):
                        idx = int(key[-1]) - 1
                        replacement_role = get_random_role(category_mapping[category][idx])
                        if not replacement_role:
                            continue
                        question = role_data[key].format(role=replacement_role)
                    else:
                        question = role_data[key].format(role=role)

                    options = [descriptor1, descriptor2]
                    random.shuffle(options)
                    pattern = "|".join(options)

                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "text", "text": question}]}
                    ]

                    try:
                        response = ask_question_single(messages=messages, pattern=pattern)
                        results.append({
                            "pair_id": pair_key,
                            "descriptor 1": descriptor1,
                            "descriptor 2": descriptor2,
                            "category": category,
                            "role": replacement_role if key.startswith("ability_irr") else role,
                            "question_id": key,
                            "question": question,
                            "response": response,
                            "options": pattern.split("|"),
                            "axis": axis,
                            "gender": gender,
                        })


                        # Save batch after every BATCH_SIZE descriptor pairs
                        if (descriptor_index + 1) % BATCH_SIZE == 0:
                            df = pd.DataFrame(results)
                            df.to_csv(output_csv_path, mode="a", header=not os.path.exists(output_csv_path), index=False)
                            pd.DataFrame({"pair_id": [r["pair_id"] for r in results]}).to_csv(progress_log_path, mode="a", header=not os.path.exists(progress_log_path), index=False)
                            # print(f"[DEBUG] Saved batch of {len(results)} responses to {output_csv_path}")
                            results = []

                    except Exception as e:
                        print(f"[ERROR] Pair {pair_key}, question {key} failed: {e}")

    # output_csv_path = os.path.join(output_dir, f"{axis}_textonly_1a_llava_responses_{gender}.csv")
    # progress_log_path = os.path.join(output_dir, f"{axis}_textonly_progress_log_{gender}.csv")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, mode="a", header=not os.path.exists(output_csv_path), index=False)
        pd.DataFrame({"pair_id": [r["pair_id"] for r in results]}).to_csv(progress_log_path, mode="a", header=not os.path.exists(progress_log_path), index=False)

print("All done!")
