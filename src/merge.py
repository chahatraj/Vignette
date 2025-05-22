import os
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm

def merge_and_save_images(image_path1, image_path2, pair_id, descriptor1, descriptor2, gender):
    # Determine the output directory
    directory = f"../../data/generated_images/noactivity_images/merged_images/{gender}/socioeconomic/"
    os.makedirs(directory, exist_ok=True)

    # Create the merged image filename
    sanitized_descriptor1 = descriptor1.replace(" ", "-").lower()
    sanitized_descriptor2 = descriptor2.replace(" ", "-").lower()
    merged_image_filename = f"{sanitized_descriptor1}--{sanitized_descriptor2}.png"
    merged_image_path = os.path.join(directory, merged_image_filename)

    # Skip if the merged image already exists
    if os.path.exists(merged_image_path):
        print(f"Skipping existing merged image: {merged_image_path}")
        return
    
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Check if images were loaded successfully
    if img1 is None:
        print(f"Error: Could not load image at {image_path1}")
        return
    if img2 is None:
        print(f"Error: Could not load image at {image_path2}")
        return

    # Ensure both images have the same height
    if img1.shape[0] != img2.shape[0]:
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (img1.shape[1], height))
        img2 = cv2.resize(img2, (img2.shape[1], height))

    # Define blending region
    blend_width = min(50, img1.shape[1], img2.shape[1])
    total_width = img1.shape[1] + img2.shape[1] - blend_width

    # Create an empty canvas for the merged image
    merged_image = np.zeros((img1.shape[0], total_width, 3), dtype=np.uint8)
    merged_image[:, :img1.shape[1] - blend_width] = img1[:, :img1.shape[1] - blend_width]
    merged_image[:, img1.shape[1]:] = img2[:, blend_width:]

    # Blend the overlapping region
    for i in range(blend_width):
        alpha = i / blend_width
        merged_image[:, img1.shape[1] - blend_width + i] = (
            (1 - alpha) * img1[:, img1.shape[1] - blend_width + i] + alpha * img2[:, i]
        ).astype(np.uint8)

    # Save the merged image
    cv2.imwrite(merged_image_path, merged_image)


def get_image_paths_and_descriptors(gender):
    base_dir = f"../../data/generated_images/noactivity_images/{gender}/socioeconomic"
    image_data = []

    if os.path.isdir(base_dir):
        print(f"Scanning directory: {base_dir}")  # Debugging print
        for file in os.listdir(base_dir):
            if file.endswith(".png"):
                base_name = file.replace(".png", "")
                descriptor = base_name  # Since no activities exist, the entire filename is the descriptor
                image_path = os.path.join(base_dir, file)
                image_data.append((image_path, descriptor))
                print(f"Found image: {image_path}")  # Debugging print

    if not image_data:
        print(f"No images found in {base_dir}")  # Debugging print

    return image_data


def merge_images_in_subdirs(gender):
    image_data = get_image_paths_and_descriptors(gender)

    if len(image_data) < 2:
        print(f"Not enough images to create pairs for {gender}.")
        return

    print(f"Merging images across descriptors for {gender}...")

    descriptor_pairs = [(img1, img2) for img1, img2 in combinations(image_data, 2)]
    
    with tqdm(total=len(descriptor_pairs), desc=f"Merging Descriptors ({gender})") as pbar:
        for (img1, img2) in descriptor_pairs:
            pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
            merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img2[1], gender)
            pbar.update(1)

# Main function to merge images for both genders
def merge_images_for_all_genders():
    for gender in ["male", "female"]:
        merge_images_in_subdirs(gender)

# Start the merging process
merge_images_for_all_genders()
