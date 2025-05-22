import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import re
import glob
import math
import matplotlib.image as mpimg

# Configuration
settings = [1]
bias_axes = [
    "ability", "age", "nationality", "race_ethnicity_color",
    "physical_traits", "religion", "socioeconomic", "gender_and_sex"
]

# Define color mappings
response_colors = {'Correct': '#AFE1AF', 'Incorrect': '#F88379'}

# Store results
aggregate_results = []
activity_results = []
all_activities = set()

# Process files and collect results
for setting, axis in itertools.product(settings, bias_axes):
    input_files = [
        f"../../outputs/factuality/flux/setting{setting}_{axis}_male_llava_responses.csv",
        f"../../outputs/factuality/flux/setting{setting}_{axis}_female_llava_responses.csv"
    ]

    combined_data = []
    for input_file in input_files:
        if os.path.exists(input_file):
            data = pd.read_csv(input_file)
            combined_data.append(data)
        else:
            print(f"Warning: {input_file} not found. Skipping.")

    if not combined_data:
        continue

    # Merge male and female datasets
    data = pd.concat(combined_data, ignore_index=True)
    data["response"] = data["response"].astype(str)

    # Normalize responses based on bias axis
    if axis != "gender_and_sex":
        data["response"] = data["response"].str.replace(r"\b(man|woman)\b", "person", flags=re.IGNORECASE, regex=True)

    if not {"activity", "response"}.issubset(data.columns):
        print(f"Error: Expected columns missing in files for {axis}. Skipping.")
        continue

    # Compute factuality score
    data["is_correct"] = data["activity"] == data["response"]
    factuality_score = data["is_correct"].mean() * 100
    print(f"Factuality test for setting {setting} on {axis}")
    print(f"Factuality Score: {factuality_score:.2f}%\n")

    # Store aggregate results
    aggregate_results.append({"Axis": axis, "Factuality Score": factuality_score})

    # Store unique activities for consistent ordering
    all_activities.update(data["activity"].unique())

    # Compute factuality scores per activity
    activity_factuality = data.groupby("activity")["is_correct"].mean() * 100
    for activity, score in activity_factuality.items():
        activity_results.append({"Axis": axis, "Activity": activity, "Factuality Score": score})

# Convert to DataFrame
agg_df = pd.DataFrame(aggregate_results).set_index("Axis")

# Plot aggregate results
plt.figure(figsize=(12, 6))
agg_df["Factuality Score"].plot(kind="bar", color="steelblue", width=0.7)
plt.title("Aggregate Factuality Scores Across Axes (Combined)", fontsize=14)
plt.ylabel("Factuality Score (%)", fontsize=12)
plt.xlabel("Bias Axis", fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.tight_layout()

# Save aggregate factuality plot
aggregate_output_file = f"../../figures/factuality/setting{settings[0]}/{axis}/{axis}_aggregate_factuality_scores_combined.png"
os.makedirs(os.path.dirname(aggregate_output_file), exist_ok=True)
plt.savefig(aggregate_output_file)
plt.show()

# Convert to DataFrame
activity_df = pd.DataFrame(activity_results)

# Process activity results per axis
all_activities = sorted(all_activities)  # Ensure consistent ordering
for axis, group in activity_df.groupby("Axis"):
    plt.figure(figsize=(20, 6))
    activity_factuality = group.set_index("Activity")["Factuality Score"].reindex(all_activities, fill_value=0)

    activity_factuality.plot(kind='bar', color='steelblue', width=0.8)
    plt.title(f"Factuality Scores by Activity\n{axis} (Setting {settings[0]})", fontsize=14)
    plt.ylabel("Factuality Score (%)", fontsize=12)
    plt.xlabel("Activity", fontsize=12)
    plt.xticks(rotation=90, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()

    # Save activity-wise factuality plot
    output_file = f"../../figures/factuality/setting{settings[0]}/{axis}/individual/{axis}_activity_factuality_combined.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()

# Descriptor Analysis
descriptor_results = []

for setting, axis in itertools.product(settings, bias_axes):
    input_files = [
        f"../../outputs/factuality/flux/setting{setting}_{axis}_male_llava_responses.csv",
        f"../../outputs/factuality/flux/setting{setting}_{axis}_female_llava_responses.csv"
    ]

    combined_data = []
    for input_file in input_files:
        if os.path.exists(input_file):
            data = pd.read_csv(input_file)
            combined_data.append(data)

    if not combined_data:
        continue

    data = pd.concat(combined_data, ignore_index=True)
    data["response"] = data["response"].astype(str)

    # Normalize responses based on bias axis
    if axis != "gender_and_sex":
        data["response"] = data["response"].str.replace(r"\b(man|woman)\b", "person", flags=re.IGNORECASE, regex=True)

    required_columns = {"question_id", "activity", "response"}
    if not required_columns.issubset(data.columns):
        continue

    # Extract descriptor from question_id
    pattern = r"What is the (.*?) doing in this image\?"
    data["Descriptor"] = data["question_id"].apply(
        lambda x: re.search(pattern, str(x)).group(1) if pd.notna(x) and re.search(pattern, str(x)) else None
    )

    data = data.dropna(subset=["Descriptor"])

    # Compute factuality score per descriptor
    data["is_correct"] = data["activity"] == data["response"]
    descriptor_factuality = data.groupby("Descriptor")["is_correct"].mean() * 100

    for descriptor, score in descriptor_factuality.items():
        descriptor_results.append({
            "Axis": axis,
            "Descriptor": descriptor,
            "Factuality Score": score
        })

# Convert to DataFrame
descriptor_df = pd.DataFrame(descriptor_results)

# # Save results
# descriptor_results_file = f"../../figs/factuality_modified/test/setting{settings[0]}/descriptor_factuality_scores_combined.csv"
# descriptor_df.to_csv(descriptor_results_file, index=False)

# print("✅ Code updated: 'man' and 'woman' are replaced with 'person' in all plots except for the 'gender_and_sex' axis.")


# Dictionary to store pivot tables for each axis
axis_pivot_dict = {}

for setting, axis in itertools.product(settings, bias_axes):
    # Same process as before to gather data:
    input_files = [
        f"../../outputs/factuality/flux/setting{setting}_{axis}_male_llava_responses.csv",
        f"../../outputs/factuality/flux/setting{setting}_{axis}_female_llava_responses.csv"
    ]

    combined_data = []
    for input_file in input_files:
        if os.path.exists(input_file):
            df_tmp = pd.read_csv(input_file)
            combined_data.append(df_tmp)

    if not combined_data:
        continue

    data = pd.concat(combined_data, ignore_index=True)
    data["response"] = data["response"].astype(str)

    # Normalize responses based on bias axis
    if axis != "gender_and_sex":
        data["response"] = data["response"].str.replace(r"\b(man|woman)\b", "person", flags=re.IGNORECASE, regex=True)

    # Check required columns
    required_columns = {"question_id", "activity", "response"}
    if not required_columns.issubset(data.columns):
        continue

    # Extract Descriptor
    pattern = r"What is the (.*?) doing in this image\?"
    data["Descriptor"] = data["question_id"].apply(
        lambda x: re.search(pattern, str(x)).group(1) if pd.notna(x) and re.search(pattern, str(x)) else None
    )
    data.dropna(subset=["Descriptor"], inplace=True)

    # Compute correctness
    data["is_correct"] = data["activity"] == data["response"]

    # Group by descriptor & activity
    descriptor_activity_df = (
        data.groupby(["Descriptor", "activity"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "Factuality Score"})
    )
    descriptor_activity_df["Factuality Score"] *= 100

    # Pivot so that rows=activities, columns=descriptors, values=Factuality Score
    pivot_df = descriptor_activity_df.pivot(index="activity", columns="Descriptor", values="Factuality Score")

    # Optionally fill any missing combinations with NaN or 0
    pivot_df = pivot_df.fillna(np.nan)

    # Store the pivot in a dictionary so we can write it all at once
    axis_pivot_dict[axis] = pivot_df

# # Write all pivot tables to a single Excel workbook, each axis on a separate sheet
# output_excel_path = f"../../figs/factuality_modified/test/setting{settings[0]}/activities_vs_descriptors.xlsx"
# with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
#     for axis, pivot_df in axis_pivot_dict.items():
#         sheet_name = str(axis)[:31]  # sheet names must be ≤31 chars
#         pivot_df.to_excel(writer, sheet_name=sheet_name)

# print(f"✅ Excel with Activities (rows) vs. Descriptors (columns) saved to {output_excel_path}")


descriptor_trend_folder = f"../../figures/factuality/setting{settings[0]}/{axis}/descriptors_trends/"
os.makedirs(descriptor_trend_folder, exist_ok=True)

for setting, axis in itertools.product(settings, bias_axes):
    print("=" * 60)
    print(f"[DEBUG] Processing Axis='{axis}', Setting={setting}")

    # Read all CSVs for this axis & setting
    input_files = [
        f"../../outputs/factuality/flux/setting{setting}_{axis}_male_llava_responses.csv",
        f"../../outputs/factuality/flux/setting{setting}_{axis}_female_llava_responses.csv"
    ]

    combined_data = []
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"[DEBUG] Reading file: {input_file}")
            df_tmp = pd.read_csv(input_file)
            print(f"[DEBUG] Shape of '{input_file}': {df_tmp.shape}")
            combined_data.append(df_tmp)
        else:
            print(f"[DEBUG] File not found: {input_file}, skipping...")

    if not combined_data:
        print(f"[DEBUG] No data found for Axis='{axis}'. Moving to next.")
        continue

    # Concatenate
    data = pd.concat(combined_data, ignore_index=True)
    print(f"[DEBUG] Combined data shape for Axis='{axis}': {data.shape}")
    print(f"[DEBUG] Columns in combined data: {list(data.columns)}")

    # Check if required columns exist
    required_columns = {"question_id", "activity", "response"}
    if not required_columns.issubset(data.columns):
        print(f"[DEBUG] Missing required columns for Axis='{axis}', skipping plot.")
        continue

    # Print a sample of question_id
    print("[DEBUG] Sample question_id values:", data["question_id"].head(3).tolist())

    # Extract Descriptor from question_id using regex
    pattern = r"What is the (.*?) doing in this image\?"
    data["Descriptor"] = data["question_id"].apply(
        lambda x: re.search(pattern, str(x)).group(1) if pd.notna(x) and re.search(pattern, str(x)) else None
    )

    # Drop rows without a valid Descriptor
    data_before_drop = data.shape[0]
    data = data.dropna(subset=["Descriptor"])
    data_after_drop = data.shape[0]
    print(f"[DEBUG] Dropped {data_before_drop - data_after_drop} rows with no valid Descriptor.")
    print(f"[DEBUG] Data shape after dropping: {data.shape}")

    # Normalize responses based on bias axis
    if axis != "gender_and_sex":
        data["response"] = data["response"].str.replace(r"\b(man|woman)\b", "person", flags=re.IGNORECASE, regex=True)

    # Compute correctness
    data["is_correct"] = data["activity"] == data["response"]

    # Group by both Descriptor and Activity, compute factuality score
    descriptor_activity_df = (
        data.groupby(["Descriptor", "activity"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "Factuality Score"})
    )
    descriptor_activity_df["Factuality Score"] *= 100

    print("[DEBUG] Head of descriptor_activity_df:")
    print(descriptor_activity_df.head())
    print(f"[DEBUG] descriptor_activity_df shape: {descriptor_activity_df.shape}")

    # Now, loop over each descriptor and plot how Factuality Score varies across activities
    grouped = descriptor_activity_df.groupby("Descriptor")
    if grouped.ngroups == 0:
        print(f"[DEBUG] No valid descriptors for Axis='{axis}', skipping plots.")
        continue

    for descriptor, group_desc in grouped:
        print(f"[DEBUG] Plotting descriptor '{descriptor}' for Axis='{axis}'")

        # Sort by activity if needed
        group_desc = group_desc.sort_values("activity")

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=group_desc,
            x="activity",
            y="Factuality Score",
            color="steelblue"
        )
        plt.title(f"Factuality by Activity\nDescriptor: '{descriptor}' | Axis: {axis} | Setting {setting}", fontsize=14)
        plt.xlabel("Activity", fontsize=12)
        plt.ylabel("Factuality Score (%)", fontsize=12)
        plt.xticks(rotation=90, ha="right")
        plt.ylim(0, 100)
        plt.tight_layout()

        # Save the figure
        plot_path = os.path.join(
            descriptor_trend_folder,
            f"{axis}_{descriptor}_factuality_by_activity.png"
        )
        plt.savefig(plot_path, dpi=300)
        plt.close()

print("✅ Descriptor vs. Activity plots have been generated and saved (if any valid data existed).")
