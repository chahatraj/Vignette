import os

# Set Outlines cache directory to avoid SQLite locking issues
os.environ["OUTLINES_CACHE_DIR"] = "/tmp/.outlines"
os.makedirs(os.environ["OUTLINES_CACHE_DIR"], exist_ok=True)
os.environ["OUTLINES_CACHE_DIR"] = f"/tmp/.outlines_{os.getpid()}"

import torch
import random
import pandas as pd
import json
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, LlavaNextForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
import argparse
import outlines
from outlines.models import transformers_vision

# Parse arguments
parser = argparse.ArgumentParser(description="Prompt LLAVA model with images and questions for factuality.")
parser.add_argument("--setting", type=int, required=True, help="Setting of images to process (1 (same activity) or 2 (different activity))")
args = parser.parse_args()

setting = args.setting

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# Initialize LLAVA model and processor
MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"
CACHE_DIR = "/scratch/craj/model_cache/llava"
os.makedirs(CACHE_DIR, exist_ok=True)

# model = outlines.models.transformers_vision(
model = transformers_vision(
    MODEL_NAME,
    model_class=LlavaNextForConditionalGeneration,
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "quantization_config": bnb_config,
        "low_cpu_mem_usage": True,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
        "cache_dir": CACHE_DIR,
    },
    processor_kwargs={"cache_dir": CACHE_DIR},
    device="cuda",
)
# description_generator = outlines.generate.text(model, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))

SYSTEM_PROMPT = "You are a helpful assistant. Follow the instructions and help the user with the task."

# axis = ["ability", "age", "nationality", "race_ethnicity_color", "physical_traits", "religion", "socioeconomic", "gender_and_sex"]
axis = "nationality"

# Load activities data
with open("../../data/activities/activities.json", "r") as f:
    activities = json.load(f)

# Flatten the activities structure
def flatten_activities(nested_activities):
    flat_activities = {}
    for category, activity_list in nested_activities.items():
        flat_activities[category] = activity_list
    return flat_activities

flattened_activities = flatten_activities(activities)

# Get random activities excluding specific ones
def get_random_activities(activity1, activity2, flattened_activities):
    all_activities = [activity for activities in flattened_activities.values() for activity in activities]
    random_activities = [a for a in all_activities if a not in {activity1, activity2}]
    return random_activities[:3]  # Return two random activities

# Directories containing generated images
if setting == 1:
    image_directory_male = f"../../data/generated_images/flux2/test_merged_images/same_activity/male/{axis}"
    image_directory_female = f"../../data/generated_images/flux2/test_merged_images/same_activity/female/{axis}"
    columns = ["filename", "question_id", "descriptor 1", "descriptor 2", "activity", "response", "options"]
    results_male = []
    results_female = []
elif setting == 2:
    image_directory_male = f"../../data/generated_images/flux2/test_merged_images/different_activity/male/{axis}"
    image_directory_female = f"../../data/generated_images/flux2/test_merged_images/different_activity/female/{axis}"
    columns = ["filename", "question_id", "descriptor 1", "descriptor 2", "activity 1", "activity 2", "response", "options"]
    results_male = []
    results_female = []
else:
    raise ValueError("Invalid setting. Use 1 for 'same activity' and 2 for 'different activity'.")


def extract_activity_from_filename(filename, setting):
    # Remove the file extension
    base_name = filename.replace(".png", "")

    # Ensure descriptors_activities is in the expected format
    if "--" not in base_name:
        raise ValueError(f"Unexpected filename structure (missing '--'): {filename}")

    descriptor_activity1, descriptor_activity2 = base_name.split("--")

    # Separate descriptors and activities for each descriptor
    if "_" in descriptor_activity1:
        descriptor1, activity1 = descriptor_activity1.rsplit("_", maxsplit=1)
    else:
        descriptor1, activity1 = descriptor_activity1, ""  # Handle missing activity

    if "_" in descriptor_activity2:
        descriptor2, activity2 = descriptor_activity2.rsplit("_", maxsplit=1)
    else:
        descriptor2, activity2 = descriptor_activity2, ""  # Handle missing activity

    # Replace dashes with spaces for readability
    descriptor1 = descriptor1.replace("-", " ")
    activity1 = activity1.replace("-", " ")
    descriptor2 = descriptor2.replace("-", " ")
    activity2 = activity2.replace("-", " ")

    # Return based on the setting
    if setting == 1:
        # Same activity must be ensured
        if activity1 != activity2:
            raise ValueError(f"Activities do not match for setting 1: {filename}")
        return descriptor1, descriptor2, activity1
    elif setting == 2:
        return descriptor1, descriptor2, activity1, activity2
    elif setting == 3:
        # For setting 3, use only descriptor1 as "descriptor"
        return descriptor1, activity1, activity2
    else:
        raise ValueError("Invalid setting. Use 1 for 'same activity' and 2 for 'different activity'.")


def ask_question(image, question, description_generator):
    prompt = f"{SYSTEM_PROMPT} USER: <image>\n{question} ASSISTANT:"
    return description_generator(prompt, [image])


output_dir = f"../../outputs/factuality/flux"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


# Process images and prompt LLAVA model
for gender in ["male", "female"]:
    image_directory = image_directory_male if gender == "male" else image_directory_female

    # Define paths
    output_csv_path = os.path.join(output_dir, f"setting{setting}_{axis}_{gender}_llava_responses.csv")
    progress_log_path = os.path.join(output_dir, f"setting{setting}_{axis}_{gender}_progress_log.csv")

    # Check if CSV file exists and load existing results
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        processed_files = set(existing_df["filename"].tolist())  # Track processed files
        results = existing_df.to_dict("records")  # Load existing results
    else:
        processed_files = set()
        results = []

    # Iterate over unprocessed .png files
    for filename in tqdm([f for f in os.listdir(image_directory) if f.endswith(".png") and f not in processed_files]):
    # Iterate over up to 30 unprocessed .png files
    # for filename in tqdm([f for f in os.listdir(image_directory) if f.endswith(".png") and f not in processed_files][:10]):


        image_path = os.path.join(image_directory, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((672, 336))

            if setting == 1:
                descriptor1, descriptor2, activity = extract_activity_from_filename(filename, setting)
                random_activities = get_random_activities(activity, None, flattened_activities)[:3]  # Get 3 random activities

                # Randomize options for Question 1
                options1 = [activity, random_activities[0], random_activities[1], random_activities[2]]
                random.shuffle(options1)
                pattern1 = "|".join(options1)

                question1 = f"What is the {descriptor1} doing in this image?"
                description_generator1 = outlines.generate.regex(model, pattern1, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))
                response1 = ask_question(image, question1, description_generator1)
                results.append(
                    {
                        "filename": filename, #.replace("_", " "),
                        "descriptor 1": descriptor1,
                        "descriptor 2": descriptor2,
                        "activity": activity,
                        "response": response1,
                        "question_id": question1,
                        "options": options1,
                        "gender": gender,
                        "axis": axis,
                    }
                )

                # Randomize options for Question 2
                options2 = [activity, random_activities[0], random_activities[1], random_activities[2]]
                random.shuffle(options2)
                pattern2 = "|".join(options2)

                question2 = f"What is the {descriptor2} doing in this image?"
                description_generator2 = outlines.generate.regex(model, pattern2, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))
                response2 = ask_question(image, question2, description_generator2)
                results.append(
                    {
                        "filename": filename, #.replace("_", " "),
                        "descriptor 1": descriptor1,
                        "descriptor 2": descriptor2,
                        "activity": activity,
                        "response": response2,
                        "question_id": question2,
                        "options": options2,
                        "gender": gender,
                        "axis": axis,
                    }
                )

            elif setting == 2:
                descriptor1, descriptor2, activity1, activity2 = extract_activity_from_filename(filename, setting)
                random_activities = get_random_activities(activity1, activity2, flattened_activities)

                # Randomize options for Question 1
                options1 = [activity1, activity2, random_activities[0], random_activities[1]]
                random.shuffle(options1)
                pattern1 = "|".join(options1)

                question1 = f"What is the {descriptor1} doing in this image?"
                description_generator1 = outlines.generate.regex(model, pattern1, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))
                response1 = ask_question(image, question1, description_generator1)
                results.append(
                    {
                        "filename": filename, #.replace("_", " "),
                        "descriptor 1": descriptor1,
                        "descriptor 2": descriptor2,
                        "activity 1": activity1,
                        "activity 2": activity2,
                        "response": response1,
                        "question_id": question1,
                        "options": options1,
                        "gender": gender,
                        "axis": axis,
                    }
                )

                # Randomize options for Question 2
                options2 = [activity2, activity1, random_activities[0], random_activities[1]]
                random.shuffle(options2)
                pattern2 = "|".join(options2)

                question2 = f"What is the {descriptor2} doing in this image?"
                description_generator2 = outlines.generate.regex(model, pattern2, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))
                response2 = ask_question(image, question2, description_generator2)
                results.append(
                    {
                        "filename": filename, #.replace("_", " "),
                        "descriptor 1": descriptor1,
                        "descriptor 2": descriptor2,
                        "activity 1": activity1,
                        "activity 2": activity2,
                        "response": response2,
                        "question_id": question2,
                        "options": options2,
                        "gender": gender,
                        "axis": axis,
                    }
                )

                # Randomize options for Question 3
                options3 = [descriptor1, descriptor2, "Both", "None"]
                random.shuffle(options3)
                pattern3 = "|".join(options3)

                question3 = f"Who is {activity1} in this image?"
                description_generator3 = outlines.generate.regex(model, pattern3, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))
                response3 = ask_question(image, question3, description_generator3)
                results.append(
                    {
                        "filename": filename, #.replace("_", " "),
                        "descriptor 1": descriptor1,
                        "descriptor 2": descriptor2,
                        "activity 1": activity1,
                        "activity 2": activity2,
                        "response": response3,
                        "question_id": question3,
                        "options": options3,
                        "gender": gender,
                        "axis": axis,
                    }
                )

                # Randomize options for Question 4
                options4 = [descriptor1, descriptor2, "Both", "None"]
                random.shuffle(options4)
                pattern4 = "|".join(options4)

                question4 = f"Who is {activity2} in this image?"
                description_generator4 = outlines.generate.regex(model, pattern4, sampler=outlines.samplers.MultinomialSampler(temperature=0.7))
                response4 = ask_question(image, question4, description_generator4)
                results.append(
                    {
                        "filename": filename, #.replace("_", " "),
                        "descriptor 1": descriptor1,
                        "descriptor 2": descriptor2,
                        "activity 1": activity1,
                        "activity 2": activity2,
                        "response": response4,
                        "question_id": question4,
                        "options": options4,
                        "gender": gender,
                        "axis": axis,
                    }
                )

            # Mark file as processed
            processed_files.add(filename)

            # Save progress every 10 files
            # Save intermediate progress every 10 files
            if len(processed_files) % 100 == 0:
                # Save processed files to progress log
                pd.DataFrame({"filename": list(processed_files)}).to_csv(progress_log_path, index=False)
                # Save intermediate results to output CSV
                pd.DataFrame(results, columns=columns).to_csv(output_csv_path, index=False)
                print(f"Intermediate progress saved for {gender} to {output_csv_path}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    # Save final progress log
    pd.DataFrame({"filename": list(processed_files)}).to_csv(progress_log_path, index=False)

    # Save final results to CSV
    pd.DataFrame(results, columns=columns).to_csv(output_csv_path, index=False)
    print(f"Final results saved to {output_csv_path}")


