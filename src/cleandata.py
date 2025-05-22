import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Function to merge descriptor files
def merge_descriptors(file_paths, output_file_path):
    dfs = [pd.read_csv(file) for file in file_paths]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.columns = ["axis", "bucket", "descriptor"]
    merged_df["axis"] = merged_df["axis"].str.lower()
    merged_df["bucket"] = merged_df["bucket"].str.lower()
    merged_df["descriptor"] = merged_df["descriptor"].str.lower()
    merged_df = merged_df.drop_duplicates(subset=["axis", "descriptor"])
    merged_df = merged_df.sort_values(by="axis")
    merged_df.to_csv(output_file_path, index=False)
    print("Merged CSV file created successfully.")

# Function to update visibility labels and calculate metrics
def update_visibility_labels(input_csv_path):
    df = pd.read_csv(input_csv_path)

    def determine_vlabel(gpt_label, human_label):
        gpt_label = gpt_label.strip() if isinstance(gpt_label, str) else gpt_label
        human_label = human_label.strip() if isinstance(human_label, str) else human_label
        if gpt_label == "No" and human_label == "No":
            return "No"
        elif gpt_label == "Yes" and human_label == "Yes":
            return "Yes"
        elif gpt_label == "Maybe" and human_label == "Maybe":
            return "Yes"
        elif (gpt_label == "Yes" and human_label == "No") or (gpt_label == "No" and human_label == "Yes"):
            return "No"
        elif (gpt_label == "Yes" and human_label == "Maybe") or (gpt_label == "Maybe" and human_label == "Yes"):
            return "Yes"
        elif gpt_label == "Maybe" and human_label == "No":
            return "No"
        elif gpt_label == "No" and human_label == "Maybe":
            return "Yes"
        else:
            print("ERROR: INPUT WRONG")
            return "ERROR"

    df["VLabel"] = df.apply(
        lambda row: determine_vlabel(row["Visibility-gpt"], row["Visibility-human"]), axis=1
    )
    df.to_csv(input_csv_path, index=False)
    print("CSV file updated successfully with VLabel column.")

    unique_axes = df["axis"].unique()
    print("Unique values in Axis column:", unique_axes)

    gpt_labels = df["Visibility-gpt"].map(lambda x: x.strip() if isinstance(x, str) else x)
    human_labels = df["Visibility-human"].map(lambda x: x.strip() if isinstance(x, str) else x)
    kappa_score = cohen_kappa_score(gpt_labels, human_labels)
    print("Cohen's Kappa between GPT and human labels:", kappa_score)

    visible_df = df[df["VLabel"] == "Yes"]
    visible_df.to_csv("../data/descriptors/merged_descriptors_visible.csv", index=False)
    print("CSV file with visible items saved successfully.")

# Updated function to create descriptor pairs for a single file
def create_descriptor_pairs_for_file(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df_yes = df[df["VLabel"] == "Yes"]
    pairs = []

    for axis, group in df_yes.groupby("axis"):
        descriptors = group["descriptor"].tolist()
        for pair in combinations(descriptors, 2):
            pairs.append([axis, str(list(pair))])

    pairs_df = pd.DataFrame(pairs, columns=["axis", "Descriptor Pair"])
    pairs_df.to_csv(output_csv_path, index=False)
    print(f"Descriptor pairs saved to {output_csv_path}")

# Updated function to filter descriptor pairs for a single file
# def filter_descriptor_pairs_for_file(input_csv_path, output_csv_path):
#     model_name = "bert-base-uncased"
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertModel.from_pretrained(model_name)
#     print("Model and tokenizer loaded successfully")

#     df = pd.read_csv(input_csv_path)
#     df["Descriptor Pair"] = df["Descriptor Pair"].apply(eval)

#     def get_word_embedding(word):
#         inputs = tokenizer(word, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     def cosine_similarity(vec1, vec2):
#         vec1 = vec1 / np.linalg.norm(vec1)
#         vec2 = vec2 / np.linalg.norm(vec2)
#         return np.dot(vec1, vec2)

#     unique_descriptors = set([desc for pair in df["Descriptor Pair"] for desc in pair])
#     descriptor_to_embedding = {
#         desc: get_word_embedding(desc).numpy() for desc in unique_descriptors
#     }

#     threshold = 0.8

#     def filter_pairs(row):
#         desc1, desc2 = row
#         if desc1 in descriptor_to_embedding and desc2 in descriptor_to_embedding:
#             vec1 = descriptor_to_embedding[desc1]
#             vec2 = descriptor_to_embedding[desc2]
#             similarity = cosine_similarity(vec1, vec2)
#             return similarity < threshold
#         return True

#     filtered_df = df[df["Descriptor Pair"].apply(filter_pairs)]
#     filtered_df.to_csv(output_csv_path, index=False)
#     print(f"Filtered descriptor pairs saved to {output_csv_path}")

# Paths for male and female descriptors
male_descriptors_csv_path = "../data/descriptors/descriptors_male.csv"
female_descriptors_csv_path = "../data/descriptors/descriptors_female.csv"

# Output paths for paired files
male_pairs_output_path = "../data/descriptors/visible_male_des_pairs.csv"
female_pairs_output_path = "../data/descriptors/visible_female_des_pairs.csv"

# Create descriptor pairs for male and female descriptors
create_descriptor_pairs_for_file(male_descriptors_csv_path, male_pairs_output_path)
create_descriptor_pairs_for_file(female_descriptors_csv_path, female_pairs_output_path)

# # Output paths for filtered files
# male_filtered_pairs_output_path = "../data/descriptors/filtered_visible_male_des_pairs.csv"
# female_filtered_pairs_output_path = "../data/descriptors/filtered_visible_female_des_pairs.csv"

# # Filter descriptor pairs for male and female descriptors
# filter_descriptor_pairs_for_file(male_pairs_output_path, male_filtered_pairs_output_path)
# filter_descriptor_pairs_for_file(female_pairs_output_path, female_filtered_pairs_output_path)
