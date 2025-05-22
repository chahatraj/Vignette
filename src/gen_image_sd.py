import os
import glob
import json
import pandas as pd
import torch
import random
from diffusers import DiffusionPipeline
from torch import autocast
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Generate images using open source model.")
parser.add_argument("--test_type", type=str, default="sampletest", help="Type of test: randomtest or sampletest")
parser.add_argument("--descriptor_file", type=str, default="male", choices=["male", "female"], required=True, help="Specify the descriptor file to use: 'male' or 'female'")
args = parser.parse_args()

# Access the arguments
test_type = str(args.test_type)
descriptor_file_choice = str(args.descriptor_file)

# Map the choice to the actual file paths
file_map = {
    "male": {
        "descriptor": "../data/descriptors/use_these_to_gen_images/maledes.csv",
    },
    "female": {
        "descriptor": "../data/descriptors/use_these_to_gen_images/femaledes.csv",
    },
}

# Determine file paths based on the user's choice
descriptor_file_path = file_map[descriptor_file_choice]["descriptor"]

# Load the specified descriptors data
descriptors_df = pd.read_csv(descriptor_file_path)
descriptor_list = descriptors_df['descriptor'].tolist()

# Load activities data
with open("../data/activities/activities.json", "r") as f:
    activities = json.load(f)

# Flatten the activities structure
def flatten_activities(nested_activities):
    flat_activities = {}
    for category, activity_list in nested_activities.items():
        flat_activities[category] = activity_list
    return flat_activities

flattened_activities = flatten_activities(activities)

# Function to get a random activity from a specific category
def get_random_activity(category):
    if category in flattened_activities:
        return random.choice(flattened_activities[category])
    else:
        raise ValueError(f"Category '{category}' not found in activities.json")

# Select descriptors and activities based on the test type
if test_type == "randomtest":
    selected_descriptors = random.sample(descriptor_list, 5)
    selected_activities = random.sample(
        [activity for activities in flattened_activities.values() for activity in activities], 5
    )
elif test_type == "sampletest":
    selected_descriptors = descriptor_list
    selected_activities = [activity for activities in flattened_activities.values() for activity in activities]

# Load the model
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    cache_dir="/scratch/craj/model_cache/flux-lora-real",
    torch_dtype=torch.float16,
)
pipeline.enable_attention_slicing()
pipeline.to("cuda")

# Log file path to track generated images
log_file_path = "../data/generated_images/generated_images_log.json"

# Load previously generated images from log file
if os.path.exists(log_file_path):
    with open(log_file_path, "r") as log_file:
        generated_images_log = json.load(log_file)
else:
    generated_images_log = {}

# Function to generate and save individual images
def generate_and_save_images(prompt, descriptor, activity):
    # Set the base directory based on the descriptor file choice
    base_directory = f"../data/generated_images/flux/individual_images/{descriptor_file_choice}/"
    os.makedirs(base_directory, exist_ok=True)

    try:
        with autocast("cuda"):
            generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000))
            image = pipeline(
                prompt, generator=generator, num_inference_steps=20
            ).images[0]
        
        # Sanitize descriptor and activity for filename
        sanitized_descriptor = descriptor.replace(" ", "-").lower()
        sanitized_activity = activity.replace(" ", "-").lower()
        
        # Combine sanitized descriptor and activity with an underscore
        filename = f"{sanitized_descriptor}_{sanitized_activity}.png"
        image_path = os.path.join(base_directory, filename)
        
        # Save the image
        image.save(image_path)
        print(f"Saved: {image_path}")
        return image_path
    except Exception as e:
        print(f"Error generating image for prompt {prompt}: {str(e)}")
        return None

# Generate individual images
individual_image_paths = {}
if test_type == "randomtest":
    for descriptor, activity in zip(selected_descriptors, selected_activities):
        key = f"{descriptor}_{activity}"
        if key in generated_images_log:
            print(f"Skipping already generated image: {key}")
            continue
        prompt = f"A {descriptor} engaged in {activity}, with their face visible."
        image_path = generate_and_save_images(prompt, descriptor=descriptor, activity=activity)
        if image_path:
            individual_image_paths[key] = image_path
            generated_images_log[key] = image_path
elif test_type == "sampletest":
    for descriptor in selected_descriptors:
        for activity in selected_activities:
            key = f"{descriptor}_{activity}"
            if key in generated_images_log:
                print(f"Skipping already generated image: {key}")
                continue
            prompt = f"A {descriptor} engaged in {activity}, with their face visible."
            image_path = generate_and_save_images(prompt, descriptor=descriptor, activity=activity)
            if image_path:
                individual_image_paths[key] = image_path
                generated_images_log[key] = image_path

# Save updated log file
with open(log_file_path, "w") as log_file:
    json.dump(generated_images_log, log_file, indent=4)

# Print generated descriptors and their image paths
print("Generated image paths:", individual_image_paths)


# # Function to generate and save individual images
# def generate_and_save_images(prompt, descriptor, activity):
#     # Set the base directory based on the descriptor file choice
#     base_directory = f"../data/generated_images/flux/individual_images/{descriptor_file_choice}/"
#     os.makedirs(base_directory, exist_ok=True)

#     # Sanitize descriptor and activity for filename
#     sanitized_descriptor = descriptor.replace(" ", "-").lower()
#     sanitized_activity = activity.replace(" ", "-").lower()
    
#     # Combine sanitized descriptor and activity with an underscore
#     filename = f"{sanitized_descriptor}_{sanitized_activity}.png"
#     image_path = os.path.join(base_directory, filename)

#     # Check if the image already exists
#     if os.path.exists(image_path):
#         print(f"Skipping already existing image: {filename}")
#         return None

#     try:
#         with autocast("cuda"):
#             generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000))
#             image = pipeline(
#                 prompt, generator=generator, num_inference_steps=20
#             ).images[0]
        
#         # Save the image
#         image.save(image_path)
#         print(f"Saved: {image_path}")
#         return image_path
#     except Exception as e:
#         print(f"Error generating image for prompt {prompt}: {str(e)}")
#         return None

# # Generate individual images
# individual_image_paths = {}
# if test_type == "randomtest":
#     for descriptor, activity in zip(selected_descriptors, selected_activities):
#         key = f"{descriptor}_{activity}"
#         prompt = f"A {descriptor} engaged in {activity}, with their face visible."
#         image_path = generate_and_save_images(prompt, descriptor=descriptor, activity=activity)
#         if image_path:
#             individual_image_paths[key] = image_path
# elif test_type == "sampletest":
#     for descriptor in selected_descriptors:
#         for activity in selected_activities:
#             key = f"{descriptor}_{activity}"
#             prompt = f"A {descriptor} engaged in {activity}, with their face visible."
#             image_path = generate_and_save_images(prompt, descriptor=descriptor, activity=activity)
#             if image_path:
#                 individual_image_paths[key] = image_path

# # Print generated descriptors and their image paths
# print("Generated image paths:", individual_image_paths)
