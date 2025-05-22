# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# from scipy.stats import fisher_exact

# # Set Seaborn theme
# sns.set_context("paper")

# # Define axis
# axis = "nationality"  # Change this to the desired axis

# # File paths
# set1_male_file = f"../../outputs/decisionmaking/llama/{axis}/{axis}_male_1a_llava_responses.csv"
# set1_female_file = f"../../outputs/decisionmaking/llama/{axis}/{axis}_female_1a_llava_responses.csv"

# set2_male_file = f"../../outputs/decisionmaking/llama/nationality_textonly/{axis}_textonly_1a_llava_responses_male.csv"
# set2_female_file = f"../../outputs/decisionmaking/llama/nationality_textonly/{axis}_textonly_1a_llava_responses_female.csv"

# fig_path = f"../../figuresforpaper/textvsvision/{axis}"
# os.makedirs(fig_path, exist_ok=True)

# axis_titles = {
#     "ability": "Ability",
#     "age": "Age",
#     "nationality": "Nationality",
#     "race_ethnicity_color": "Race/Ethnicity",
#     "physical_traits": "Physical Traits",
#     "religion": "Religion",
#     "socioeconomic": "Socioeconomic",
#     "gender_and_sex": "Gender and Sex"
# }

# def load_and_process(files):
#     dataframes = []
#     for file in files:
#         if os.path.exists(file):
#             df = pd.read_csv(file)
#             dataframes.append(df)
#         else:
#             print(f"Warning: {file} not found. Skipping.")

#     if not dataframes:
#         return None

#     combined_df = pd.concat(dataframes, ignore_index=True)

#     # def to_gender_neutral(descriptor):
#     #     descriptor = str(descriptor)
#     #     descriptor = re.sub(r'\b(male|female)\b', '', descriptor, flags=re.IGNORECASE)
#     #     descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
#     #     descriptor = re.sub(r'\b(filipino|filipina|filipana|Philippines|nan Philippines)\b', 'Philippines', descriptor, flags=re.IGNORECASE)
#     #     descriptor = re.sub(r'\b(filipino|filipina)\b', 'Philippines person', descriptor, flags=re.IGNORECASE)
#     #     return descriptor.strip()

#     def to_gender_neutral(descriptor):
#         descriptor = str(descriptor).strip().lower()
#         descriptor = descriptor.replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ')
#         descriptor = re.sub(r'\s+', ' ', descriptor)  # collapse multiple spaces

#         # Fix malformed "nan" variants
#         if re.match(r'^nan\s*filip.*', descriptor):
#             descriptor = "philippines person"

#         # Remove gendered terms
#         descriptor = re.sub(r'\b(male|female)\b', '', descriptor)
#         descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor)

#         # Normalize all filipino variants
#         descriptor = re.sub(r'\b(filipino|filipina|filipana)\b', 'philippines person', descriptor)
#         descriptor = descriptor.replace("philippines", "philippines person")  # unify all

#         return descriptor.strip()


#     # print("Unique values in descriptor 1:")
#     # print(combined_df['descriptor 1'].dropna().unique())

#     # print("Unique values in descriptor 2:")
#     # print(combined_df['descriptor 2'].dropna().unique())

#     # print("Unique values in response:")
#     # print(combined_df['response'].dropna().unique())

#     # Apply normalization
#     for col in ['descriptor 1', 'descriptor 2', 'response']:
#         combined_df[col] = combined_df[col].apply(to_gender_neutral)

#     # Step 2: Drop any remaining nan-based garbage rows
#     combined_df = combined_df.dropna(subset=['descriptor 1', 'descriptor 2', 'response'])

#     # Drop rows where descriptor columns still contain literal "nan"
#     for col in ['descriptor 1', 'descriptor 2', 'response']:
#         combined_df = combined_df[~combined_df[col].str.contains("nan", case=False)]


#     # combined_df['descriptor 1'] = combined_df['descriptor 1'].apply(to_gender_neutral)
#     # combined_df['descriptor 2'] = combined_df['descriptor 2'].apply(to_gender_neutral)
#     # combined_df['response'] = combined_df['response'].apply(to_gender_neutral)

#     descriptor_counts = pd.concat([combined_df['descriptor 1'], combined_df['descriptor 2']]).value_counts()
#     response_counts = combined_df['response'].value_counts()
#     adjusted_response_percentage = (response_counts / descriptor_counts) * 100

#     response_df = adjusted_response_percentage.reset_index()
#     response_df.columns = ["Descriptor", "Adjusted Response Percentage"]
#     response_df = response_df.sort_values(by="Descriptor", ascending=True)

#     fisher_p_values = {}
#     total_responses = response_counts.sum()
#     total_options = descriptor_counts.sum()
#     for desc in response_df["Descriptor"]:
#         x1 = response_counts.get(desc, 0)
#         n1 = descriptor_counts.get(desc, 0)
#         x2 = total_responses - x1
#         n2 = total_options - n1

#         table = np.array([[x1, n1 - x1],
#                           [x2, n2 - x2]])
#         if table.min() >= 0 and table.sum() > 0:
#             try:
#                 _, p_value = fisher_exact(table)
#             except Exception:
#                 p_value = np.nan
#         else:
#             p_value = np.nan
#         fisher_p_values[desc] = p_value

#     response_df["p_value"] = response_df["Descriptor"].map(fisher_p_values)
#     response_df["Clean Descriptor"] = response_df["Descriptor"].str.replace("person", "", regex=False).str.strip()
#     return response_df

# # Process both sets
# response_df_1 = load_and_process([set1_male_file, set1_female_file])
# response_df_2 = load_and_process([set2_male_file, set2_female_file])

# # Merge both sets on cleaned descriptor
# merged_df = pd.merge(response_df_1, response_df_2, on="Clean Descriptor", how="outer", suffixes=("_set1", "_set2"))
# merged_df = merged_df.fillna(0)  # Fill missing values with 0

# # Sort and setup for plotting
# merged_df = merged_df.sort_values("Adjusted Response Percentage_set1", ascending=True)
# y_base = np.arange(len(merged_df))
# offset = 0.15

# # Plot
# height = max(4, len(merged_df) * 0.5)
# plt.figure(figsize=(13, height), constrained_layout=True)

# # Set 1 (left stick)
# for x, y, p in zip(merged_df["Adjusted Response Percentage_set1"], y_base - offset, merged_df["p_value_set1"]):
#     plt.hlines(y=y, xmin=0, xmax=x, color='teal', linewidth=2.5)
#     marker = "*" if p < 0.05 else "o"
#     plt.plot(x, y, marker, markersize=14 if marker == "*" else 10, color='teal')

# # Set 2 (right stick)
# for x, y, p in zip(merged_df["Adjusted Response Percentage_set2"], y_base + offset, merged_df["p_value_set2"]):
#     plt.hlines(y=y, xmin=0, xmax=x, color='#C3B1E1', linewidth=2.5)
#     marker = "*" if p < 0.05 else "o"
#     plt.plot(x, y, marker, markersize=14 if marker == "*" else 10, color='#C3B1E1')

# # Labels
# plt.yticks(y_base, merged_df["Clean Descriptor"], fontsize=12, fontweight='medium')
# xmax = max(merged_df["Adjusted Response Percentage_set1"].max(),
#            merged_df["Adjusted Response Percentage_set2"].max())
# plt.xlim(0, xmax * 1.1)
# plt.xticks(fontsize=12, fontweight='medium')
# plt.title(f"{axis_titles.get(axis, axis)}", fontsize=16, weight='bold')

# # Style
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(2)
#     spine.set_edgecolor('black')

# # Legend
# plt.plot([], [], color='teal', marker='o', linestyle='-', label='Text+Vision')
# plt.plot([], [], color='#C3B1E1', marker='o', linestyle='-', label='Text-only')
# plt.legend(loc='lower right')

# # Save
# plt.savefig(os.path.join(fig_path, f"text_vs_vision_{axis}_sidebyside_lollipop.png"), dpi=300, bbox_inches="tight")
# plt.show()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm 
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
import re

# -----------------------------
# Configuration
# -----------------------------
axis = "nationality"
set1_files = [
    f"../../outputs/decisionmaking/llama/{axis}/{axis}_male_1a_llava_responses.csv",
    f"../../outputs/decisionmaking/llama/{axis}/{axis}_female_1a_llava_responses.csv"
]
set2_files = [
    f"../../outputs/decisionmaking/llama/nationality_textonly/{axis}_textonly_1a_llava_responses_male.csv",
    f"../../outputs/decisionmaking/llama/nationality_textonly/{axis}_textonly_1a_llava_responses_female.csv"
]
fig_path = f"../../figuresforpaper/textvsvision/{axis}"
os.makedirs(fig_path, exist_ok=True)

# -----------------------------
# Colormap for Delta
# -----------------------------
vlag_r = plt.get_cmap("vlag_r")
spectral = plt.get_cmap("Spectral")

# Blend the two colormaps (e.g., 60% vlag_r, 40% Spectral)
vlag_colors = vlag_r(np.linspace(0, 1, 256))
spectral_colors = spectral(np.linspace(0, 1, 256))
blended_colors = 0.6 * vlag_colors + 0.4 * spectral_colors

# Create new colormap
blended_cmap = ListedColormap(blended_colors)

# Reverse the blended colormap
blended_colors_r = blended_colors[::-1]

# Create new colormap
blended_cmap_r = ListedColormap(blended_colors_r)


# Load colormaps
# Define your custom hex colors
# colors = ["#C3B1E1", "#008080"]

# Create the custom colormap
# delta_cmap = LinearSegmentedColormap.from_list("PurpleTeal", colors, N=256)

# colors = ["#C3B1E1", "#E6E6E6", "#008080"]
colors = ["#bda5e6", "#E6E6E6", "#006363"]
delta_cmap = LinearSegmentedColormap.from_list("LavenderTealDiverge", colors, N=256)

# Optional: reverse the blended colormap
delta_cmap_r = ListedColormap(blended_colors[::-1])

# color_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

# -----------------------------
# Gender-Neutral Cleanup Function
# -----------------------------
def to_gender_neutral(descriptor):
    descriptor = str(descriptor).strip().lower()
    descriptor = descriptor.replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ')
    descriptor = re.sub(r'\s+', ' ', descriptor)
    if re.match(r'^nan\s*filip.*', descriptor):
        descriptor = "philippines person"
    descriptor = re.sub(r'\b(male|female)\b', '', descriptor)
    descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor)
    descriptor = re.sub(r'\b(filipino|filipina|filipana)\b', 'philippines person', descriptor)
    descriptor = descriptor.replace("philippines", "philippines person")
    return descriptor.strip()

# -----------------------------
# Data Loading & Cleaning
# -----------------------------
def load_and_process(files):
    dfs = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"Warning: {file} not found.")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    for col in ['descriptor 1', 'descriptor 2', 'response']:
        df[col] = df[col].apply(to_gender_neutral)
    df = df.dropna(subset=['descriptor 1', 'descriptor 2', 'response'])
    for col in ['descriptor 1', 'descriptor 2', 'response']:
        df = df[~df[col].str.contains("nan", case=False)]
    return df

# -----------------------------
# Convert to Response Matrix
# -----------------------------
def get_response_matrix(df):
    if {'question_id', 'response', 'category'}.issubset(df.columns):
        ability_irr_ids = ['ability_irr1', 'ability_irr2', 'ability_irr3']
        ability_df = df[df['question_id'].isin(ability_irr_ids)].copy()
        ability_agg = ability_df.groupby(['category', 'response']).size().unstack(fill_value=0)
        ability_agg.index = ['ability (' + cat + ')' for cat in ability_agg.index]
        non_ability_df = df[~df['question_id'].isin(ability_irr_ids)]
        non_agg = non_ability_df.groupby(['question_id', 'response']).size().unstack(fill_value=0)
        return pd.concat([non_agg, ability_agg], axis=0)
    else:
        return df.groupby(['question_id', 'response']).size().unstack(fill_value=0)

# -----------------------------
# Process Data
# -----------------------------
df1 = load_and_process(set1_files)
df2 = load_and_process(set2_files)

# Clean up response matrix indices BEFORE subtraction
mat1 = get_response_matrix(df1)
mat2 = get_response_matrix(df2)

# Remove 'person', lowercase, strip for both matrices BEFORE delta
mat1.index = mat1.index.str.replace("person", "", regex=False).str.strip().str.lower()
mat2.index = mat2.index.str.replace("person", "", regex=False).str.strip().str.lower()
mat1.columns = mat1.columns.str.replace("person", "", regex=False).str.strip().str.lower()
mat2.columns = mat2.columns.str.replace("person", "", regex=False).str.strip().str.lower()


mat1 = mat1.div(mat1.sum(axis=1), axis=0) * 100
mat2 = mat2.div(mat2.sum(axis=1), axis=0) * 100

# Align rows and columns
all_rows = mat1.index.union(mat2.index)
all_cols = mat1.columns.union(mat2.columns)
mat1 = mat1.reindex(index=all_rows, columns=all_cols).fillna(0)
mat2 = mat2.reindex(index=all_rows, columns=all_cols).fillna(0)

# -----------------------------
# Compute Delta and Plot
# -----------------------------
delta = mat1 - mat2  # Δ = Text+Vision - Text-only
# Filter out ability_irr rows
delta_filtered = delta.loc[delta.index.str.fullmatch("ability") | ~delta.index.str.startswith("ability (")]

# -----------------------------
# Run Z-test for significance
# -----------------------------
raw_mat1 = get_response_matrix(df1)
raw_mat2 = get_response_matrix(df2)

# Clean up indices and columns
raw_mat1.index = raw_mat1.index.str.replace("person", "", regex=False).str.strip().str.lower()
raw_mat2.index = raw_mat2.index.str.replace("person", "", regex=False).str.strip().str.lower()
raw_mat1.columns = raw_mat1.columns.str.replace("person", "", regex=False).str.strip().str.lower()
raw_mat2.columns = raw_mat2.columns.str.replace("person", "", regex=False).str.strip().str.lower()

# Align shapes
raw_mat1 = raw_mat1.reindex(index=delta.index, columns=delta.columns).fillna(0)
raw_mat2 = raw_mat2.reindex(index=delta.index, columns=delta.columns).fillna(0)

# Compute Z-test p-values
pval_df = pd.DataFrame(index=delta.index, columns=delta.columns)
for i in delta.index:
    for j in delta.columns:
        x1 = raw_mat1.loc[i, j]
        x2 = raw_mat2.loc[i, j]
        n1 = raw_mat1.loc[i].sum()
        n2 = raw_mat2.loc[i].sum()

        if n1 == 0 or n2 == 0:
            pval_df.loc[i, j] = np.nan
            continue

        p1 = x1 / n1
        p2 = x2 / n2
        p = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        if se == 0:
            pval_df.loc[i, j] = np.nan
        else:
            z = (p1 - p2) / se
            pval_df.loc[i, j] = 2 * (1 - norm.cdf(abs(z)))

# -----------------------------
# Prepare annotations for heatmap (bold only)
# -----------------------------
annotations = delta_filtered.round(1).astype(str)
for i in delta_filtered.index:
    for j in delta_filtered.columns:
        p = pval_df.loc[i, j]
        if pd.notna(p) and p < 0.05:
            annotations.loc[i, j] = f"$\\bf{{{annotations.loc[i, j]}}}$"


abs_max = np.abs(delta_filtered.values).max()
color_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
# color_norm = TwoSlopeNorm(vmin=-0.75 * abs_max, vcenter=0, vmax=0.75 * abs_max)

# -----------------------------
# Plot heatmap with significance (manual hearts)
# -----------------------------
plt.figure(figsize=(max(26, 0.4 * len(delta_filtered.columns)), max(8, 0.4 * len(delta_filtered.index))))
ax = sns.heatmap(delta_filtered, annot=annotations, fmt="", cmap=delta_cmap, norm=color_norm,
                 linewidths=1, linecolor='black', cbar_kws={"pad": 0.02}, annot_kws={"size": 16})

# Add hearts manually
for i, row in enumerate(delta_filtered.index):
    for j, col in enumerate(delta_filtered.columns):
        pval = pval_df.loc[row, col]
        if pd.notna(pval) and pval < 0.05:
            ax.annotate('♥', xy=(j + 0.7, i + 0.2), xytext=(8, 8),
                        textcoords='offset points', color='black',
                        fontsize=18, ha='center', va='top', annotation_clip=False)

# Style ticks and labels
xticklabels = [label.get_text().replace(" person", "").strip() for label in ax.get_xticklabels()]
yticklabels = [label.get_text().replace(" person", "").strip() for label in ax.get_yticklabels()]
if axis == "nationality":
    xticklabels = [label.replace("native american", "native amer.").replace("middle eastern", "middle east") for label in xticklabels]
    yticklabels = [label.replace("native american", "native amer.").replace("middle eastern", "middle east") for label in yticklabels]
    
ax.set_xticklabels(xticklabels, rotation=90, ha='right', fontsize=25, fontweight='medium')
ax.set_yticklabels(yticklabels, rotation=0, fontsize=25, fontweight='medium')
ax.tick_params(axis='x', width=2, pad=5)
ax.tick_params(axis='y', width=2)
ax.collections[0].colorbar.ax.tick_params(labelsize=25, width=3)

# Style borders
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(2)

# Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"{axis}_delta_heatmap_significance.png"), dpi=600, bbox_inches='tight')
plt.show()

print(f"✅ Saved Δ significance heatmap at: {os.path.join(fig_path, f'{axis}_delta_heatmap_significance.png')}")




# plt.figure(figsize=(10, 8))
# sns.heatmap(delta_filtered.T, cmap=delta_cmap, center=0, annot=True, fmt=".1f",
#             linewidths=0.5, cbar_kws={'label': 'Δ Response % (Vision - Text)'})

# plt.title(f"Δ Heatmap: Text+Vision vs Text-only ({axis.title()})", fontsize=16, weight='bold')
# plt.xlabel("")
# plt.ylabel("")

# save_path = os.path.join(fig_path, f"{axis}_delta_heatmap.png")
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# plt.show()

# print(f"✅ Saved Δ heatmap at: {save_path}")




# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import seaborn as sns
# import os
# import json
# import re
# import numpy as np

# # Load the reversed vlag colormap
# vlag_r = plt.get_cmap("vlag_r")
# spectral = plt.get_cmap("Spectral")

# # Blend the two colormaps (e.g., 60% vlag_r, 40% Spectral)
# vlag_colors = vlag_r(np.linspace(0, 1, 256))
# spectral_colors = spectral(np.linspace(0, 1, 256))
# blended_colors = 0.6 * vlag_colors + 0.4 * spectral_colors

# # Create new colormap
# blended_cmap = ListedColormap(blended_colors)

# # Define the axis
# # # axis = ["ability", "age", "nationality", "race_ethnicity_color", "physical_traits", "religion", "socioeconomic", "gender_and_sex"]
# axis = "nationality"

# # File paths
# male_file = f"../../outputs/decisionmaking/llama/{axis}/{axis}_male_1a_llava_responses.csv"
# female_file = f"../../outputs/decisionmaking/llama/{axis}/{axis}_female_1a_llava_responses.csv"
# json_file = "../../data/activities/roles.json"
# fig_path = f"../../figuresforpaper/textvsvision/{axis}"

# os.makedirs(fig_path, exist_ok=True)

# # Load data if files exist
# dataframes = []
# for file in [male_file, female_file]:
#     if os.path.exists(file):
#         df = pd.read_csv(file)
#         dataframes.append(df)
#     else:
#         print(f"Warning: {file} not found. Skipping.")

# # Combine data
# if dataframes:
#     combined_df = pd.concat(dataframes, ignore_index=True)

#     def to_gender_neutral(descriptor):
#         """Convert gendered descriptor to gender-neutral format."""
#         descriptor = re.sub(r'\b(male|female)\b', '', descriptor, flags=re.IGNORECASE)
#         descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
#         descriptor = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', descriptor, flags=re.IGNORECASE)
#         return descriptor

#     # Apply gender-neutral conversion
#     combined_df['descriptor 1'] = combined_df['descriptor 1'].apply(to_gender_neutral)
#     combined_df['descriptor 2'] = combined_df['descriptor 2'].apply(to_gender_neutral)
#     combined_df['response'] = combined_df['response'].apply(to_gender_neutral)
    
#     # Compute descriptor frequencies in options (descriptor 1 and descriptor 2)
#     descriptor_counts = pd.concat([combined_df['descriptor 1'], combined_df['descriptor 2']]).value_counts()
    
#     # Count occurrences of each descriptor in the response column
#     response_counts = combined_df['response'].value_counts()
    
#     # Calculate adjusted response percentage
#     adjusted_response_percentage = (response_counts / descriptor_counts) * 100
    
#     # Convert to DataFrame for Seaborn plotting
#     response_df = adjusted_response_percentage.reset_index()
#     response_df.columns = ["Descriptor", "Adjusted Response Percentage"]
    
#     # Load category-role mapping from JSON
#     with open(json_file, 'r') as f:
#         role_categories = json.load(f)

#     # Create a mapping of roles to categories
#     role_to_category = {}
#     for category, roles in role_categories.items():
#         for role_info in roles:
#             role_to_category[role_info['role']] = category

#     # Add category column to the dataframe
#     combined_df['category'] = combined_df['role'].map(role_to_category)
    
#     # Drop rows with missing categories
#     combined_df = combined_df.dropna(subset=['category'])
    
#     # Define the required category order
#     category_order = ["Necessary Time", "Contracted Time", "Committed Time", "Free Time"]
    
#     # Sort roles alphabetically within each category
#     sorted_roles = []
#     category_labels = []
#     for category in category_order:
#         if category in role_categories:
#             category_roles = sorted([role_info['role'] for role_info in role_categories[category]])
#             sorted_roles.extend(category_roles + [None])  # Add a None placeholder for whitespace
#             category_labels.extend([category] * len(category_roles) + [None])
    
#     # Remove trailing None placeholder
#     if sorted_roles[-1] is None:
#         sorted_roles.pop()
#         category_labels.pop()
    
#     # Reorder response_role_percentages based on sorted roles
#     response_role_counts = combined_df.groupby(['role', 'response']).size().unstack(fill_value=0)
#     response_role_percentages = response_role_counts.div(response_role_counts.sum(axis=1), axis=0) * 100
#     response_role_percentages = response_role_percentages.reindex(sorted_roles)
    
#     # Replace None rows with NaN for whitespace in heatmap
#     response_role_percentages.loc[None] = np.nan
    
#     # Determine figure size dynamically
#     num_roles = len(response_role_percentages.index)
#     num_responses = len(response_role_percentages.columns)

#     plt.figure(figsize=(num_roles * 0.2, num_responses * 0.2))

#     # Plot heatmap with responses on y-axis and roles on x-axis
#     sns.heatmap(response_role_percentages.T, cmap=blended_cmap, annot=False, linewidths=0.5, cbar=True, mask=response_role_percentages.T.isna(), cbar_kws={"pad": 0.01})

#     plt.xlabel("")
#     plt.ylabel("")
#     plt.title("")
#     plt.xticks(rotation=90, ha="center")
#     plt.yticks(rotation=0)
    
#     # Extract category positions
#     category_positions = []
#     prev_category = None
#     for idx, role in enumerate(sorted_roles):
#         category = category_labels[idx]
#         if category != prev_category and category is not None:
#             category_positions.append((idx, category))
#         prev_category = category
    
#     # Add category labels centered for each category
#     for i in range(len(category_positions)):
#         start_pos, category = category_positions[i]
#         if i < len(category_positions) - 1:
#             end_pos = category_positions[i + 1][0]
#         else:
#             end_pos = len(sorted_roles)  # Ensure Free Time category is included
#         center_pos = (start_pos + end_pos) / 2
#         plt.subplots_adjust(bottom=-0.05)  # Increase bottom margin
#         plt.text(center_pos, -4, category, ha='center', va='bottom', fontsize=10, color='black', rotation=0)
    
#     plt.tight_layout()

#     # Save plot
#     plot_path_heatmap = os.path.join(fig_path, f"{axis}_roles_vs_responses_heatmap.png")
#     os.makedirs(os.path.dirname(plot_path_heatmap), exist_ok=True)
#     plt.savefig(plot_path_heatmap, bbox_inches='tight')
#     plt.close()
    
#     print(f"Categorized heatmap saved successfully at {plot_path_heatmap}.")