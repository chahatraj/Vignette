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
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from scipy.stats import fisher_exact, kruskal

sns.set_context("paper")

# Load the reversed vlag colormap
vlag_r = plt.get_cmap("vlag_r")
spectral = plt.get_cmap("Spectral")

# Blend the two colormaps (e.g., 60% vlag_r, 40% Spectral)
vlag_colors = vlag_r(np.linspace(0, 1, 256))
spectral_colors = spectral(np.linspace(0, 1, 256))
blended_colors = 0.6 * vlag_colors + 0.4 * spectral_colors

# Create new colormap
blended_cmap = ListedColormap(blended_colors)

# Configuration
settings = [1]
# bias_axes = [
#     "ability", "age", "nationality", "race_ethnicity_color",
#     "physical_traits", "religion", "socioeconomic", "gender_and_sex"
# ]
bias_axes = ["gender_and_sex"]


# # Store results
# aggregate_results = []
# activity_results = []
# all_activities = set()

# # Process files and collect results
# for setting, axis in itertools.product(settings, bias_axes):
#     input_files = [
#         f"../../outputs/factuality/flux/setting{setting}_{axis}_male_llava_responses.csv",
#         f"../../outputs/factuality/flux/setting{setting}_{axis}_female_llava_responses.csv"
#     ]

#     combined_data = []
#     for input_file in input_files:
#         if os.path.exists(input_file):
#             data = pd.read_csv(input_file)
#             combined_data.append(data)
#         else:
#             print(f"Warning: {input_file} not found. Skipping.")

#     if not combined_data:
#         continue

#     # Merge male and female datasets
#     data = pd.concat(combined_data, ignore_index=True)
#     data["response"] = data["response"].astype(str)

#     # Normalize responses based on bias axis
#     if axis != "gender_and_sex":
#         data["response"] = data["response"].str.replace(r"\b(man|woman)\b", "person", flags=re.IGNORECASE, regex=True)

#     if not {"activity", "response"}.issubset(data.columns):
#         print(f"Error: Expected columns missing in files for {axis}. Skipping.")
#         continue

#     # Compute factuality score
#     data["is_correct"] = data["activity"] == data["response"]
#     factuality_score = data["is_correct"].mean() * 100
#     print(f"Factuality test for setting {setting} on {axis}")
#     print(f"Factuality Score: {factuality_score:.2f}%\n")

#     # Store aggregate results
#     aggregate_results.append({"Axis": axis, "Factuality Score": factuality_score})

#     # Store unique activities for consistent ordering
#     all_activities.update(data["activity"].unique())

#     # Compute factuality scores per activity
#     activity_factuality = data.groupby("activity")["is_correct"].mean() * 100
#     for activity, score in activity_factuality.items():
#         activity_results.append({"Axis": axis, "Activity": activity, "Factuality Score": score})

# # Convert to DataFrame
# agg_df = pd.DataFrame(aggregate_results).set_index("Axis")

# # Use a color palette for the bars
# colors = sns.color_palette("magma", len(agg_df))

# plt.figure(figsize=(12, 6))
# agg_df["Factuality Score"].plot(kind="bar", color=colors, width=0.7)
# plt.title("Aggregate Factuality Scores Across Axes (Combined)", fontsize=14)
# plt.ylabel("Factuality Score (%)", fontsize=12)
# plt.xlabel("Bias Axis", fontsize=12)
# plt.xticks(rotation=45)
# plt.ylim(0, 100)
# plt.tight_layout()

# # Save aggregate factuality plot
# aggregate_output_file = f"../../figures/factuality/setting{settings[0]}/aggregate_factuality_scores_combined.png"
# os.makedirs(os.path.dirname(aggregate_output_file), exist_ok=True)
# plt.savefig(aggregate_output_file)
# plt.show()

# # Convert to DataFrame
# activity_df = pd.DataFrame(activity_results)

# # Process activity results per axis
# # Pivot the activity DataFrame for heatmap representation
# heatmap_data = activity_df.pivot(index="Axis", columns="Activity", values="Factuality Score")

# # Create the heatmap
# plt.figure(figsize=(50, 8))
# sns.heatmap(heatmap_data, annot=True, cmap=blended_cmap, cbar=True, cbar_kws={"pad": 0.02}, linewidths=0.5, vmin=0, vmax=100, fmt=".1f")

# # Formatting
# plt.title(f"Factuality Scores Heatmap (Setting {settings[0]})", fontsize=16)
# plt.xlabel("Activity", fontsize=12)
# plt.ylabel("Bias Axis", fontsize=12)
# plt.xticks(rotation=90, ha="right")
# plt.yticks(rotation=0)
# plt.tight_layout()

# # Save the heatmap
# heatmap_output_file = f"../../figures/factuality/setting{settings[0]}/activity_factuality_heatmap.png"
# os.makedirs(os.path.dirname(heatmap_output_file), exist_ok=True)
# plt.savefig(heatmap_output_file, format="png", dpi=300, bbox_inches="tight")
# plt.show()



#plot 2 - heatmaps
# Store results
activity_results = []
all_activities = set()

# Process files and collect results
for setting, axis in itertools.product(settings, bias_axes):
    input_files = [
        f"../../../backupresults/outputs/factuality/flux/setting{setting}_{axis}_male_llava_responses.csv",
        f"../../../backupresults/outputs/factuality/flux/setting{setting}_{axis}_female_llava_responses.csv"
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

    # Extract unique descriptors **for this axis only**
    def to_gender_neutral(descriptor, axis):
        """Convert gendered descriptor to gender-neutral format, except for gender_and_sex axis."""
        if axis != "gender_and_sex":
            descriptor = re.sub(r'\b(male|female|boy|girl)\b', '', descriptor, flags=re.IGNORECASE)
            descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', descriptor, flags=re.IGNORECASE)
        return descriptor.strip()


    def plot_lollipop(descriptor_avg_df, axis, setting_num):
        """
        Generate a colorful lollipop plot showing factuality scores per descriptor.
        """
        # plt.figure(figsize=(max(6, len(descriptor_avg_df) * 0.3), 4))
        # plt.figure(figsize=(8, max(4, len(descriptor_avg_df) * 0.4)))
        height = 2 if axis == "gender_and_sex" else max(4, len(descriptor_avg_df) * 0.4)
        plt.figure(figsize=(6, height), constrained_layout=True)

        descriptor_avg_df = descriptor_avg_df.sort_values("Factuality Score", ascending=True)
        y_pos = np.arange(len(descriptor_avg_df))
        if axis == "gender_and_sex":
            # y_pos = np.linspace(0, 0.1, len(descriptor_avg_df))
            y_pos = np.linspace(0, len(descriptor_avg_df) - 1, len(descriptor_avg_df))


        # Normalize scores for colormap (0–1)
        norm_scores = descriptor_avg_df["Factuality Score"] / 100
        # colors = cm.viridis(norm_scores)
        # colors = cm.Set2(1 - norm_scores)
        # colors = ['teal'] * len(descriptor_avg_df)
        colors = ['teal' if p < 0.05 else 'gray' for p in descriptor_avg_df["p_value"]]




        # Plot colorful lines
        # for i, (x, y, c) in enumerate(zip(descriptor_avg_df["Factuality Score"], y_pos, colors)):
        #     plt.hlines(y=y, xmin=0, xmax=x, color=c, alpha=0.8, linewidth=2)
        #     plt.plot(x, y, "o", markersize=10, color=c)
        for i, (x, y, c, p) in enumerate(zip(descriptor_avg_df["Factuality Score"], y_pos, colors, descriptor_avg_df["p_value"])):
            plt.hlines(y=y, xmin=0, xmax=x, color=c, alpha=0.8, linewidth=2.5)
            marker = "*" if p < 0.05 else "o"
            plt.plot(x, y, marker, markersize=14 if marker == "*" else 10, color=c)


        # # Add value labels
        # for x, y in zip(descriptor_avg_df["Factuality Score"], y_pos):
        #     plt.text(x + 2, y, f"{x:.1f}%", va='center', fontsize=22)

        axis_titles = {
            "ability": "Ability",
            "age": "Age",
            "nationality": "Nationality",
            "race_ethnicity_color": "Race/Ethnicity",
            "physical_traits": "Physical Traits",
            "religion": "Religion",
            "socioeconomic": "Socioeconomic",
            "gender_and_sex": "Gender and Sex"
        }


        # Formatting
        descriptor_avg_df["Descriptor"] = descriptor_avg_df["Descriptor"].str.replace("person", "", regex=False).str.strip()
        # plt.yticks(y_pos, descriptor_avg_df["Descriptor"])
        plt.yticks(y_pos, descriptor_avg_df["Descriptor"], fontsize=14, fontweight='medium')
        if axis == "gender_and_sex":
            plt.ylim(-0.5, len(descriptor_avg_df) - 0.5)
        # plt.xlabel("Factuality Score (%)", fontsize=14)
        # plt.title(f"Factuality Scores of Descriptors ({axis})", fontsize=18, weight='bold')
        plt.title(f"{axis_titles.get(axis, axis)}", fontsize=20, weight='bold')
        plt.xlim(0, 100)
        plt.xticks(fontsize=14, fontweight='medium')
        # plt.grid(axis='x', linestyle='--', alpha=0.4)
        # plt.grid(axis='x', linestyle='--', linewidth=1.5, alpha=0.7, color='gray')

        # plt.tight_layout()

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')


        # Save the lollipop plot
        lollipop_output_file = f"../../figuresforpaper/factuality/setting{setting_num}/{axis}_descriptors_lollipop.png"
        os.makedirs(os.path.dirname(lollipop_output_file), exist_ok=True)
        plt.savefig(lollipop_output_file, format="png", dpi=300, bbox_inches="tight")
        plt.show()



    # Extract and normalize descriptors from `question_id`
    pattern = r"What is the (.*?) doing in this image\?"
    # data["Descriptor"] = data["question_id"].astype(str).apply(
    #     lambda x: to_gender_neutral(re.search(pattern, x).group(1)) if re.search(pattern, x) else None
    # )
    data["Descriptor"] = data["question_id"].astype(str).apply(
        lambda x: to_gender_neutral(re.search(pattern, x).group(1), axis) if re.search(pattern, x) else None
    )


    # Drop NaN descriptors
    data = data.dropna(subset=["Descriptor"])

    # Normalize responses based on bias axis
    if axis != "gender_and_sex":
        data["response"] = data["response"].str.replace(r"\b(male|female)\b", "", flags=re.IGNORECASE, regex=True)
        data["response"] = data["response"].str.replace(r"\b(man|woman|boy|girl)\b", "person", flags=re.IGNORECASE, regex=True)
        data["response"] = data["response"].str.replace(r"\b(filipino|filipina)\b", "phillipines person", flags=re.IGNORECASE, regex=True)


    if not {"activity", "response"}.issubset(data.columns):
        print(f"Error: Expected columns missing in files for {axis}. Skipping.")
        continue

    # Compute factuality score per descriptor
    # Compute factuality score per descriptor-activity pair
    data["is_correct"] = data["activity"] == data["response"]
    descriptor_factuality = data.groupby(["Descriptor", "activity"])["is_correct"].mean() * 100


    # Store results
    descriptor_results = []
    for (descriptor, activity), score in descriptor_factuality.items():
        descriptor_results.append({
            "Axis": axis,
            "Descriptor": descriptor,
            "Activity": activity,
            "Factuality Score": score
        })

    # Convert to DataFrame
    descriptor_df = pd.DataFrame(descriptor_results)

    # **Modify Heatmap Data to Include Descriptor-Activity Factuality**
    heatmap_data = descriptor_df.pivot(index="Descriptor", columns="Activity", values="Factuality Score")

    # Ensure all extracted descriptors are included in the heatmap
    unique_descriptors = sorted(data["Descriptor"].unique())  # Ensures all descriptors are included
    heatmap_data = heatmap_data.reindex(index=unique_descriptors, fill_value=np.nan)

    # **Generate Heatmap for Descriptors vs Activities**
    # plt.figure(figsize=(20, 8))
    plt.figure(figsize=(max(50, len(heatmap_data.columns) * 0.5), max(6, len(heatmap_data.index) * 0.3)))
    sns.heatmap(heatmap_data, annot=True, cmap=blended_cmap, cbar=True, cbar_kws={"pad": 0.02}, linewidths=0.5, vmin=0, vmax=100, fmt=".1f")

    # Formatting
    plt.title(f"Factuality Scores Heatmap)", fontsize=16)
    plt.xlabel("", fontsize=12)
    plt.ylabel(f"", fontsize=12)
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)

    # Save heatmap
    heatmap_output_file = f"../../figuresforpaper/factuality/setting{settings[0]}/{axis}_descriptors_vs_activity_heatmap.png"
    os.makedirs(os.path.dirname(heatmap_output_file), exist_ok=True)
    plt.savefig(heatmap_output_file, format="png", dpi=300, bbox_inches="tight")
    plt.show()

#   plot 3 - barplots

    # Compute factuality score per descriptor (averaged across all activities)
    descriptor_factuality_avg = data.groupby("Descriptor")["is_correct"].mean() * 100

    # Get raw counts for each descriptor
    descriptor_counts = data.groupby("Descriptor")["is_correct"].agg(['sum', 'count'])
    descriptor_counts.rename(columns={'sum': 'correct', 'count': 'total'}, inplace=True)
    descriptor_counts["incorrect"] = descriptor_counts["total"] - descriptor_counts["correct"]

    # Kruskal-Wallis test (global across all descriptors)
    grouped_corrects = [data[data["Descriptor"] == desc]["is_correct"].astype(int) 
                        for desc in descriptor_counts.index]
    kruskal_stat, kruskal_p = kruskal(*grouped_corrects)
    print(f"Kruskal-Wallis p-value for {axis}: {kruskal_p:.4f}")

    # Fisher's exact test (per descriptor vs. rest)
    fisher_p_values = {}
    for desc in descriptor_counts.index:
        x1 = descriptor_counts.loc[desc, 'correct']
        n1 = descriptor_counts.loc[desc, 'total']
        x2 = descriptor_counts['correct'].sum() - x1
        n2 = descriptor_counts['total'].sum() - n1

        table = np.array([[x1, n1 - x1],
                        [x2, n2 - x2]])
        if table.min() >= 0 and table.sum() > 0:
            try:
                _, p_value = fisher_exact(table)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan
        fisher_p_values[desc] = p_value

    # # Attach p-values to descriptor_avg_df
    # descriptor_avg_df["p_value"] = descriptor_avg_df["Descriptor"].map(fisher_p_values)


    # Convert to DataFrame
    descriptor_avg_df = pd.DataFrame({
        "Descriptor": descriptor_factuality_avg.index,
        "Factuality Score": descriptor_factuality_avg.values
    })

    # Attach p-values to descriptor_avg_df
    descriptor_avg_df["p_value"] = descriptor_avg_df["Descriptor"].map(fisher_p_values)

    # **Generate Bar Plot for Descriptors' Factuality Scores**
    plt.figure(figsize=(max(6, len(descriptor_avg_df) * 0.3), 3))
    sns.barplot(x="Descriptor", y="Factuality Score", data=descriptor_avg_df, palette="magma")

    # Formatting
    plt.title(f"Factuality Scores of Descriptors ({axis})", fontsize=16)
    plt.xlabel("Descriptor", fontsize=12)
    plt.ylabel("Factuality Score (%)", fontsize=12)
    plt.xticks(rotation=90, ha="right")
    plt.ylim(0, 100)

    # Save the bar plot
    barplot_output_file = f"../../figuresforpaper/factuality/setting{settings[0]}/{axis}_descriptors_barplot.png"
    os.makedirs(os.path.dirname(barplot_output_file), exist_ok=True)
    plt.savefig(barplot_output_file, format="png", dpi=300, bbox_inches="tight")
    plt.show()

    plot_lollipop(descriptor_avg_df, axis, settings[0])
