import os
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm
import argparse

def sanitize(text):
    return text.replace(" ", "-").lower()

def merge_and_save_images(path1, path2, merged_image_path):
    if os.path.exists(merged_image_path):
        return

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print(f"Error loading: {path1} or {path2}")
        return

    blend_width = min(50, img1.shape[1], img2.shape[1])
    total_width = img1.shape[1] + img2.shape[1] - blend_width
    merged_image = np.zeros((img1.shape[0], total_width, 3), dtype=np.uint8)

    # Copy non-blended regions
    merged_image[:, :img1.shape[1] - blend_width] = img1[:, :img1.shape[1] - blend_width]
    merged_image[:, img1.shape[1]:] = img2[:, blend_width:]

    # Vectorized blend
    alpha = np.linspace(0, 1, blend_width).reshape(1, blend_width, 1)
    blended = ((1 - alpha) * img1[:, -blend_width:] + alpha * img2[:, :blend_width]).astype(np.uint8)
    merged_image[:, img1.shape[1] - blend_width:img1.shape[1]] = blended

    cv2.imwrite(merged_image_path, merged_image)



def get_image_data(gender, axis):
    base_dir = f"../individual_images/{gender}/{axis}"
    image_data = []

    if not os.path.isdir(base_dir):
        return image_data

    for file in os.listdir(base_dir):
        if file.endswith(".png") and "_" in file:
            descriptor, activity = file[:-4].split("_", 1)
            sanitized_descriptor = sanitize(descriptor)
            sanitized_activity = sanitize(activity)
            path = os.path.join(base_dir, file)
            image_data.append({
                "descriptor": descriptor,
                "activity": activity,
                "sanitized_name": f"{sanitized_descriptor}_{sanitized_activity}",
                "path": path
            })
    return image_data

def merge_images_by_type(gender, image_data, merge_type, condition_fn, axis):
    out_dir = f"../{merge_type}/{gender}/{axis}"
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "merged.log")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            already_done = set(line.strip() for line in f)
    else:
        already_done = set()

    pairs = [
        (img1, img2)
        for img1, img2 in combinations(image_data, 2)
        if condition_fn(img1, img2)
    ]

    with tqdm(total=len(pairs), desc=f"{merge_type.replace('_', ' ').title()}") as pbar, open(log_path, "a") as log_file:
        for img1, img2 in pairs:
            merged_name = f"{img1['sanitized_name']}--{img2['sanitized_name']}.png"
            merged_path = os.path.join(out_dir, merged_name)

            if merged_name not in already_done:
                merge_and_save_images(img1["path"], img2["path"], merged_path)
                log_file.write(merged_name + "\n")
            pbar.update(1)


def merge_images_in_subdirs(gender, contrast_types, axis):
    image_data = get_image_data(gender, axis)
    if len(image_data) < 2:
        print(f"Not enough images for gender: {gender}")
        return

    if "identity_contrast" in contrast_types:
        merge_images_by_type(
            gender,
            image_data,
            "identity_contrast",
            lambda x, y: x["descriptor"] != y["descriptor"] and x["activity"] == y["activity"],
            axis
        )

    if "activity_identity_contrast" in contrast_types:
        merge_images_by_type(
            gender,
            image_data,
            "activity_identity_contrast",
            lambda x, y: x["descriptor"] != y["descriptor"] and x["activity"] != y["activity"],
            axis
        )

    if "activity_contrast" in contrast_types:
        merge_images_by_type(
            gender,
            image_data,
            "activity_contrast",
            lambda x, y: x["descriptor"] == y["descriptor"] and x["activity"] != y["activity"],
            axis
        )

def merge_images_for_all_genders(contrast_types, axis):
    for gender in ["male", "female"]:
        merge_images_in_subdirs(gender, contrast_types, axis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["identity_contrast", "activity_identity_contrast", "activity_contrast", "all"],
        default=["all"],
        help="Specify which contrast types to run"
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=[
            "ability",
            "age",
            "gender_and_sex",
            "nationality",
            "physical_traits",
            "race_ethnicity_color",
            "religion",
            "socioeconomic"
        ],
        default="age",
        help="Axis to process"
    )
    args = parser.parse_args()

    contrast_types = (
        ["identity_contrast", "activity_identity_contrast", "activity_contrast"]
        if "all" in args.types else args.types
    )

    merge_images_for_all_genders(contrast_types, args.axis)







# import os
# import cv2
# import numpy as np
# from itertools import combinations
# from tqdm import tqdm

# def merge_and_save_images(image_path1, image_path2, pair_id, descriptor1, activity1, descriptor2, activity2, gender, merge_type):
#     # Determine the output directory based on the merge type
#     directory = f"../../data/generated_images/flux2/test_merged_images/{merge_type}/{gender}/nationality/"
#     os.makedirs(directory, exist_ok=True)

#     # Create the merged image filename
#     sanitized_descriptor1 = descriptor1.replace(" ", "-").lower()
#     sanitized_activity1 = activity1.replace(" ", "-").lower()
#     sanitized_descriptor2 = descriptor2.replace(" ", "-").lower()
#     sanitized_activity2 = activity2.replace(" ", "-").lower()
#     merged_image_filename = f"{sanitized_descriptor1}_{sanitized_activity1}--{sanitized_descriptor2}_{sanitized_activity2}.png"
#     merged_image_path = os.path.join(directory, merged_image_filename)

#     # Add this check to skip if the merged image already exists
#     if os.path.exists(merged_image_path):
#         print(f"Skipping existing merged image: {merged_image_path}")
#         return
    
#     img1 = cv2.imread(image_path1)
#     img2 = cv2.imread(image_path2)

#     # Check if images were loaded successfully
#     if img1 is None:
#         print(f"Error: Could not load image at {image_path1}")
#         return
#     if img2 is None:
#         print(f"Error: Could not load image at {image_path2}")
#         return

#     # Ensure both images have the same height
#     if img1.shape[0] != img2.shape[0]:
#         height = min(img1.shape[0], img2.shape[0])
#         img1 = cv2.resize(img1, (img1.shape[1], height))
#         img2 = cv2.resize(img2, (img2.shape[1], height))

#     # Define blending region
#     blend_width = min(50, img1.shape[1], img2.shape[1])
#     total_width = img1.shape[1] + img2.shape[1] - blend_width

#     # Create an empty canvas for the merged image
#     merged_image = np.zeros((img1.shape[0], total_width, 3), dtype=np.uint8)
#     merged_image[:, :img1.shape[1] - blend_width] = img1[:, :img1.shape[1] - blend_width]
#     merged_image[:, img1.shape[1]:] = img2[:, blend_width:]

#     # Blend the overlapping region
#     for i in range(blend_width):
#         alpha = i / blend_width
#         merged_image[:, img1.shape[1] - blend_width + i] = (
#             (1 - alpha) * img1[:, img1.shape[1] - blend_width + i] + alpha * img2[:, i]
#         ).astype(np.uint8)

#     # Determine the output directory based on the merge type
#     directory = f"../../data/generated_images/flux2/test_merged_images/{merge_type}/{gender}/nationality/"
#     os.makedirs(directory, exist_ok=True)

#     # Sanitize descriptors and activities for filename
#     sanitized_descriptor1 = descriptor1.replace(" ", "-").lower()
#     sanitized_activity1 = activity1.replace(" ", "-").lower()
#     sanitized_descriptor2 = descriptor2.replace(" ", "-").lower()
#     sanitized_activity2 = activity2.replace(" ", "-").lower()

#     # Create the merged image filename
#     merged_image_filename = f"{sanitized_descriptor1}_{sanitized_activity1}--{sanitized_descriptor2}_{sanitized_activity2}.png"
#     merged_image_path = os.path.join(directory, merged_image_filename)
#     cv2.imwrite(merged_image_path, merged_image)


# def get_image_paths_and_descriptors(gender):
#     base_dir = f"../../data/generated_images/flux/individual_images/{gender}/"
#     image_data = []

#     # Process only the 'nationality' axis
#     axis = "nationality"
#     axis_dir = os.path.join(base_dir, axis)
#     if os.path.isdir(axis_dir):
#         for file in os.listdir(axis_dir):
#             if file.endswith(".png"):
#                 base_name = file.replace(".png", "")
#                 if "_" in base_name:
#                     descriptor, activity = base_name.split("_", 1)
#                     image_path = os.path.join(axis_dir, file)
#                     image_data.append((image_path, descriptor, activity, axis))

#     return image_data

# def merge_images_in_subdirs(gender):
#     image_data = get_image_paths_and_descriptors(gender)

#     if len(image_data) < 2:
#         print(f"Not enough images to create pairs for {gender}.")
#         return

#     # Group images by axis (only 'nationality')
#     axis_groups = {"nationality": image_data}

#     # Merge images within the 'nationality' axis
#     for axis, images in axis_groups.items():
#         print(f"Merging images in '{axis}' for {gender}...")

#         # Same Activity, Different Descriptor
#         same_activity_pairs = [
#             (img1, img2) for img1, img2 in combinations(images, 2) if img1[1] != img2[1] and img1[2] == img2[2]
#         ]
#         with tqdm(total=len(same_activity_pairs), desc=f"Same Activity ({axis})") as pbar:
#             for (img1, img2) in same_activity_pairs:
#                 pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
#                 merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img1[2], img2[1], img2[2], gender, "same_activity")
#                 pbar.update(1)
#                 # break

#         # Different Activity, Different Descriptor
#         different_activity_pairs = [
#             (img1, img2) for img1, img2 in combinations(images, 2) if img1[1] != img2[1] and img1[2] != img2[2]
#         ]
#         with tqdm(total=len(different_activity_pairs), desc=f"Different Activity ({axis})") as pbar:
#             for (img1, img2) in different_activity_pairs:
#                 # merge_and_save_images(*img1, *img2, gender, "different_activity")
#                 pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
#                 merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img1[2], img2[1], img2[2], gender, "different_activity")
#                 pbar.update(1)
#                 # break

#         # Same Descriptor, Different Activity
#         same_descriptor_pairs = [
#             (img1, img2) for img1, img2 in combinations(images, 2) if img1[1] == img2[1] and img1[2] != img2[2]
#         ]
#         with tqdm(total=len(same_descriptor_pairs), desc=f"Same Descriptor ({axis})") as pbar:
#             for (img1, img2) in same_descriptor_pairs:
#                 # merge_and_save_images(*img1, *img2, gender, "same_descriptor")
#                 pair_id = f"{os.path.basename(img1[0]).replace('.png', '')}--{os.path.basename(img2[0]).replace('.png', '')}"
#                 merge_and_save_images(img1[0], img2[0], pair_id, img1[1], img1[2], img2[1], img2[2], gender, "same_descriptor")
#                 pbar.update(1)
#                 # break

# # Main function to merge images for both genders
# def merge_images_for_all_genders():
#     for gender in ["male"]:
#         merge_images_in_subdirs(gender)

# # Start the merging process
# merge_images_for_all_genders()