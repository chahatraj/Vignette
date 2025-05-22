import os
import json
import pandas as pd
import torch
import random
from diffusers import DiffusionPipeline
from torch import autocast
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Generate images using an open-source model without activity.")
parser.add_argument("--test_type", type=str, default="sampletest", help="Type of test: randomtest or sampletest")
parser.add_argument("--descriptor_file", type=str, choices=["male", "female"], required=True, help="Specify the descriptor file to use: 'male' or 'female'")
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

# Select descriptors based on the test type
if test_type == "randomtest":
    selected_descriptors = random.sample(descriptor_list, 5)
elif test_type == "sampletest":
    selected_descriptors = descriptor_list

# Load the model
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    cache_dir="/scratch/craj/model_cache/flux-lora-real",
    torch_dtype=torch.float16,
)
pipeline.enable_attention_slicing()
pipeline.to("cuda")

# Log file path to track generated images
log_file_path = "../data/generated_images/generated_noactivity_images_log.json"

# Load previously generated images from log file
if os.path.exists(log_file_path):
    with open(log_file_path, "r") as log_file:
        generated_images_log = json.load(log_file)
else:
    generated_images_log = {}

# Function to generate and save individual images
def generate_and_save_images(prompt, descriptor):
    # Set the base directory based on the descriptor file choice
    base_directory = f"../data/generated_images/noactivity_images/{descriptor_file_choice}/"
    os.makedirs(base_directory, exist_ok=True)

    try:
        with autocast("cuda"):
            generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000))
            image = pipeline(
                prompt, generator=generator, num_inference_steps=20
            ).images[0]
        
        # Sanitize descriptor for filename
        sanitized_descriptor = descriptor.replace(" ", "-").lower()
        
        # Filename based only on descriptor
        filename = f"{sanitized_descriptor}.png"
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
    for descriptor in selected_descriptors:
        if descriptor in generated_images_log:
            print(f"Skipping already generated image: {descriptor}")
            continue
        prompt = f"A {descriptor}, with their face visible."
        image_path = generate_and_save_images(prompt, descriptor=descriptor)
        if image_path:
            individual_image_paths[descriptor] = image_path
            generated_images_log[descriptor] = image_path
elif test_type == "sampletest":
    for descriptor in selected_descriptors:
        if descriptor in generated_images_log:
            print(f"Skipping already generated image: {descriptor}")
            continue
        prompt = f"A {descriptor}, with their face visible."
        image_path = generate_and_save_images(prompt, descriptor=descriptor)
        if image_path:
            individual_image_paths[descriptor] = image_path
            generated_images_log[descriptor] = image_path

# Save updated log file
with open(log_file_path, "w") as log_file:
    json.dump(generated_images_log, log_file, indent=4)

# Print generated descriptors and their image paths
print("Generated image paths:", individual_image_paths)
