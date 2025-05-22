import os
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm

def merge_and_save_images(image_path1, image_path2, pair_id, descriptor1, activity1, descriptor2, activity2, gender, merge_type):
    # Determine the output directory based on the merge type
    directory = f"../../data/generated_images/flux2/test_merged_images/{merge_type}/{gender}/nationality/"
    os.makedirs(directory, exist_ok=True)

    # Create the merged image filename
    sanitized_descriptor1 = descriptor1.replace(" ", "-").lower()
    sanitized_activity1 = activity1.replace(" ", "-").lower()
    sanitized_descriptor2 = descriptor2.replace(" ", "-").lower()
    sanitized_activity2 = activity2.replace(" ", "-").lower()
    merged_image_filename = f"{sanitized_descriptor1}_{sanitized_activity1}--{sanitized_descriptor2}_{sanitized_activity2}.png"
    merged_image_path = os.path.join(directory, merged_image_filename)

    # Add this check to skip if the merged image already exists
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

    # Determine the output directory based on the merge type
    directory = f"../../data/generated_images/flux2/test_merged_images/{merge_type}/{gender}/nationality/"
    os.makedirs(directory, exist_ok=True)

    # Sanitize descriptors and activities for filename
    sanitized_descriptor1 = descriptor1.replace(" ", "-").lower()
    sanitized_activity1 = activity1.replace(" ", "-").lower()
    sanitized_descriptor2 = descriptor2.replace(" ", "-").lower()
    sanitized_activity2 = activity2.replace(" ", "-").lower()

    # Create the merged image filename
    merged_image_filename = f"{sanitized_descriptor1}_{sanitized_activity1}--{sanitized_descriptor2}_{sanitized_activity2}.png"
    merged_image_path = os.path.join(directory, merged_image_filename)
    cv2.imwrite(merged_image_path, merged_image)


def get_image_paths_and_descriptors(gender):
    base_dir = f"../../data/generated_images/flux/individual_images/{gender}/"
    image_data = []

    # Process only the 'nationality' axis
    axis = "nationality"
    axis_dir = os.path.join(base_dir, axis)
    if os.path.isdir(axis_dir):
        for file in os.listdir(axis_dir):
            if file.endswith(".png"):
                base_name = file.replace(".png", "")
                if "_" in base_name:
                    descriptor, activity = base_name.split("_", 1)
                    image_path = os.path.join(axis_dir, file)
                    image_data.append((image_path, descriptor, activity, axis))

    return image_data

def merge_images_in_subdirs(gender):
    image_data = get_image_paths_and_descriptors(gender)

    if len(image_data) < 2:
        print(f"Not enough images to create pairs for {gender}.")
        return

    # Group images by axis (only 'nationality')
    axis_groups = {"nationality": image_data}

    # Merge images within the 'nationality' axis
    for axis, images in axis_groups.items():
        print(f"Merging images in '{axis}' for {gender}...")

        # Same Activity, Different Descriptor
        same_activity_pairs = [
            (img1, img2) for img1, img2 in combinations(images, 2) if img1[1] != img2[1] and img1[2] == img2[2]
        ]
        with tqdm(total=len(same_activity_pairs), desc=f"Same Activity ({axis})") as pbar:
            for (img1, img2) in same_activity_pairs:
                pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
                merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img1[2], img2[1], img2[2], gender, "same_activity")
                pbar.update(1)
                # break

        # Different Activity, Different Descriptor
        different_activity_pairs = [
            (img1, img2) for img1, img2 in combinations(images, 2) if img1[1] != img2[1] and img1[2] != img2[2]
        ]
        with tqdm(total=len(different_activity_pairs), desc=f"Different Activity ({axis})") as pbar:
            for (img1, img2) in different_activity_pairs:
                # merge_and_save_images(*img1, *img2, gender, "different_activity")
                pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
                merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img1[2], img2[1], img2[2], gender, "different_activity")
                pbar.update(1)
                # break

        # Same Descriptor, Different Activity
        same_descriptor_pairs = [
            (img1, img2) for img1, img2 in combinations(images, 2) if img1[1] == img2[1] and img1[2] != img2[2]
        ]
        with tqdm(total=len(same_descriptor_pairs), desc=f"Same Descriptor ({axis})") as pbar:
            for (img1, img2) in same_descriptor_pairs:
                # merge_and_save_images(*img1, *img2, gender, "same_descriptor")
                pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
                merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img1[2], img2[1], img2[2], gender, "same_descriptor")
                pbar.update(1)
                # break

# Main function to merge images for both genders
def merge_images_for_all_genders():
    for gender in ["male"]:
        merge_images_in_subdirs(gender)

# Start the merging process
merge_images_for_all_genders()