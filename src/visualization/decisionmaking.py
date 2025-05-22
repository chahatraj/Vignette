# # PLOT 1 - BARPLOT
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re

# # Set Seaborn theme for better visuals
# sns.set_context("paper")
# palette = sns.color_palette("flare", as_cmap=True)  # Using a perceptually uniform colormap

# # Define the axis
# # # axis = ["ability", "age", "nationality", "race_ethnicity_color", "physical_traits", "religion", "socioeconomic", "gender_and_sex"]
# axis = "ability"

# # File paths
# male_file = f"../../outputs/decisionmaking/flux/{axis}/{axis}_male_1a_llava_responses.csv"
# female_file = f"../../outputs/decisionmaking/flux/{axis}/{axis}_female_1a_llava_responses.csv"
# fig_path = f"../../figuresforpaper/decisionmaking/flux/{axis}"

# # Make directories if not exist
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
#         descriptor = str(descriptor)
#         descriptor = re.sub(r'\b(male|female)\b', '', descriptor, flags=re.IGNORECASE)
#         descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
#         descriptor = re.sub(r'\b(filipino|filipina)\b', 'Philippines person', descriptor, flags=re.IGNORECASE)
#         return descriptor.strip()

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
#     # response_df = response_df.sort_values(by="Adjusted Response Percentage", ascending=False)
#     response_df = response_df.sort_values(by="Descriptor", ascending=True)

#     # Plot using Seaborn (Vertical Bars)
#     plt.figure(figsize=(14, 6))  # Wider figure for better spacing
#     sns.barplot(
#         x="Descriptor",
#         y="Adjusted Response Percentage",
#         data=response_df,
#         palette="flare",  # Enhanced color palette
#         edgecolor="black"
#     )

#     # Formatting the plot
#     plt.ylabel("", fontsize=12, fontweight="bold")
#     plt.xlabel("", fontsize=12, fontweight="bold")
#     plt.title(f"{axis}", fontsize=14, fontweight="bold")
#     plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate for readability
#     plt.yticks(fontsize=10)

#     # Save and show the plot
#     plt.savefig(os.path.join(fig_path, f"{axis}_response_percentage.png"), dpi=300, bbox_inches="tight")
#     plt.show()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import fisher_exact

# Set Seaborn theme
sns.set_context("paper")

# Define axis
axis = "nationality"  # Change this to the desired axis

# File paths
male_file = f"../../outputs/decisionmaking/llama/{axis}/{axis}_male_1a_llava_responses.csv"
female_file = f"../../outputs/decisionmaking/llama/{axis}/{axis}_female_1a_llava_responses.csv"

male_file = f"../../outputs/decisionmaking/llama/nationality_textonly/{axis}_textonly_1a_llava_responses_male.csv"
female_file = f"../../outputs/decisionmaking/llama/nationality_textonly/{axis}_textonly_1a_llava_responses_female.csv"
fig_path = f"../../figuresforpaper/textvsvision/{axis}"

# Make directories if not exist
os.makedirs(fig_path, exist_ok=True)

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

# Load data
dataframes = []
for file in [male_file, female_file]:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dataframes.append(df)
    else:
        print(f"Warning: {file} not found. Skipping.")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    def to_gender_neutral(descriptor):
        descriptor = str(descriptor)
        descriptor = re.sub(r'\b(male|female)\b', '', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(filipino|filipina)\b', 'Philippines person', descriptor, flags=re.IGNORECASE)
        return descriptor.strip()

    # Apply gender-neutral conversion
    combined_df['descriptor 1'] = combined_df['descriptor 1'].apply(to_gender_neutral)
    combined_df['descriptor 2'] = combined_df['descriptor 2'].apply(to_gender_neutral)
    combined_df['response'] = combined_df['response'].apply(to_gender_neutral)

    # Compute descriptor frequencies in options
    descriptor_counts = pd.concat([combined_df['descriptor 1'], combined_df['descriptor 2']]).value_counts()

    # Count occurrences in response
    response_counts = combined_df['response'].value_counts()

    # Calculate adjusted response percentage
    adjusted_response_percentage = (response_counts / descriptor_counts) * 100

    # Prepare DataFrame
    response_df = adjusted_response_percentage.reset_index()
    response_df.columns = ["Descriptor", "Adjusted Response Percentage"]
    response_df = response_df.sort_values(by="Descriptor", ascending=True)

    # Calculate Fisher's exact p-values
    fisher_p_values = {}
    total_responses = response_counts.sum()
    total_options = descriptor_counts.sum()
    for desc in response_df["Descriptor"]:
        x1 = response_counts.get(desc, 0)
        n1 = descriptor_counts.get(desc, 0)
        x2 = total_responses - x1
        n2 = total_options - n1

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

    response_df["p_value"] = response_df["Descriptor"].map(fisher_p_values)

    # --- Lollipop Plot ---
    height = max(4, len(response_df) * 0.4)
    plt.figure(figsize=(12, height), constrained_layout=True)

    response_df = response_df.sort_values("Adjusted Response Percentage", ascending=True)
    y_pos = np.arange(len(response_df))

    # Color by significance
    colors = ['teal' if p < 0.05 else 'gray' for p in response_df["p_value"]]

    for i, (x, y, c, p) in enumerate(zip(response_df["Adjusted Response Percentage"], y_pos, colors, response_df["p_value"])):
        plt.hlines(y=y, xmin=0, xmax=x, color=c, alpha=0.8, linewidth=2.5)
        marker = "*" if p < 0.05 else "o"
        plt.plot(x, y, marker, markersize=14 if marker == "*" else 10, color=c)

    # Clean "person" from display labels
    response_df["Clean Descriptor"] = response_df["Descriptor"].str.replace("person", "", regex=False).str.strip()

    # Plot
    plt.yticks(y_pos, response_df["Clean Descriptor"], fontsize=12, fontweight='medium')

    plt.xlim(0, response_df["Adjusted Response Percentage"].max() * 1.1)
    plt.xticks(fontsize=12, fontweight='medium')
    plt.title(f"{axis_titles.get(axis, axis)}", fontsize=16, weight='bold')


    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Save plot
    plt.savefig(os.path.join(fig_path, f"text_{axis}_response_lollipop.png"), dpi=300, bbox_inches="tight")
    plt.show()




 
# #PLOT 2 - HEATMAP
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import ListedColormap
# import re

# # Set Seaborn theme
# sns.set_context("paper")

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
# axis = "gender_and_sex"

# # File paths
# male_file = f"../../outputs/decisionmaking/flux/{axis}/{axis}_male_1a_llava_responses.csv"
# female_file = f"../../outputs/decisionmaking/flux/{axis}/{axis}_female_1a_llava_responses.csv"
# fig_path = f"../../figures/decisionmaking/{axis}"

# # Ensure directory exists
# os.makedirs(fig_path, exist_ok=True)

# # Load data
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
#         descriptor = str(descriptor)
#         # descriptor = re.sub(r'\b(male|female)\b', '', descriptor, flags=re.IGNORECASE)
#         # descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
#         descriptor = re.sub(r'\b(filipino|filipina)\b', 'Philippines person', descriptor, flags=re.IGNORECASE)
#         return descriptor.strip()

#     # Apply gender-neutral conversion
#     combined_df['descriptor 1'] = combined_df['descriptor 1'].apply(to_gender_neutral)
#     combined_df['descriptor 2'] = combined_df['descriptor 2'].apply(to_gender_neutral)
#     combined_df['response'] = combined_df['response'].apply(to_gender_neutral)

#     # **Handling ability_irr separately**
#     if {'question_id', 'response', 'category'}.issubset(combined_df.columns):
#         # Identify all ability_irr question IDs
#         ability_irr_ids = ['ability_irr1', 'ability_irr2', 'ability_irr3']
        
#         # Filter ability_irr related rows
#         ability_irr_df = combined_df[combined_df['question_id'].isin(ability_irr_ids)].copy()

#         # Aggregate ability_irr responses based on their category
#         ability_aggregated = ability_irr_df.groupby(['category', 'response']).size().unstack(fill_value=0)

#         # Rename index to formatted ability labels
#         ability_aggregated.index = ['ability (' + cat + ')' for cat in ability_aggregated.index]

#         # Get all other question_id response counts
#         non_ability_df = combined_df[~combined_df['question_id'].isin(ability_irr_ids)]
#         non_ability_counts = non_ability_df.groupby(['question_id', 'response']).size().unstack(fill_value=0)

#         # Combine both datasets
#         final_response_counts = pd.concat([non_ability_counts, ability_aggregated], axis=0)
#     else:
#         print("🚨 Required columns missing! Using only regular question IDs.")
#         final_response_counts = combined_df.groupby(['question_id', 'response']).size().unstack(fill_value=0)

#     # Normalize to response percentages
#     final_response_counts = final_response_counts.div(final_response_counts.sum(axis=1), axis=0) * 100

#     # **Now pivot and plot heatmap using the fixed dataset**
#     plt.figure(figsize=(12, 5))
#     # sns.heatmap(final_response_counts, cmap=blended_cmap, annot=True, fmt=".1f", linewidths=0.5, cbar_kws={'label': 'Response Percentage'})
#     ax = sns.heatmap(final_response_counts.loc[final_response_counts.index.str.fullmatch("ability") | 
#                                       ~final_response_counts.index.str.startswith("ability (")].T, 
#             cmap=blended_cmap, annot=True, fmt=".1f", linewidths=0.5, cbar_kws={'label': ''})

#     new_labels = [
#         label.get_text().replace(" person", "\nperson") if "down syndrome" not in label.get_text() and "college student" not in label.get_text() 
#         else label.get_text().replace("down syndrome person", "down\nsyndrome\nperson").replace("college student", "college\nstudent") 
#         for label in ax.get_xticklabels()
#     ]
#     ax.set_xticklabels(new_labels, ha="center")


#     plt.ylabel("", fontsize=12, fontweight="bold")
#     plt.xlabel("", fontsize=12, fontweight="bold")
#     plt.title("", fontsize=14, fontweight="bold")

#     # Save heatmap
#     heatmap_save_path = os.path.join(fig_path, f"{axis}_heatmap_response_percentage.png")
#     plt.savefig(heatmap_save_path, dpi=300, bbox_inches="tight")

#     # Show plot
#     plt.show()

#     print(f"✅ Saved heatmap at: {heatmap_save_path}")






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
# male_file = f"../../outputs/decisionmaking/flux/{axis}/{axis}_male_1a_llava_responses.csv"
# female_file = f"../../outputs/decisionmaking/flux/{axis}/{axis}_female_1a_llava_responses.csv"
# json_file = "../../data/activities/roles.json"
# fig_path = f"../../figures/decisionmaking/{axis}"

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


