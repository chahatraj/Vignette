import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import re
from scipy.stats import fisher_exact

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

# Define the axis
# # axis = ["ability", "age", "nationality", "race_ethnicity_color", "physical_traits", "religion", "socioeconomic", "gender_and_sex"]
axis = "race_ethnicity_color"  # Change this to the desired axis

# File paths
# male_file = f"../../outputs/socialassumptions/flux/{axis}/{axis}_male_1a_llava_responses.csv"
# female_file = f"../../outputs/socialassumptions/flux/{axis}/{axis}_female_1a_llava_responses.csv"

# flux
# male_file = f"../../../backupresults/outputs/socialassumptions/flux/{axis}/{axis}_male_1a_llava_responses.csv"
# female_file = f"../../../backupresults/outputs/socialassumptions/flux/{axis}/{axis}_female_1a_llava_responses.csv"
# fig_path_net_score = f"../../figuresforpaper/socialassumptions/flux/{axis}"

#llama
male_file = f"../../outputs/socialassumptions/llama/{axis}/{axis}_male_1a_llava_responses.csv"
female_file = f"../../outputs/socialassumptions/llama/{axis}/{axis}_female_1a_llava_responses.csv"
fig_path_net_score = f"../../figuresforpaper/socialassumptions/llama/{axis}"

# # #deepseek
# male_file = f"../../outputs/socialassumptions/deepseek/{axis}/{axis}_male_1a_deepseek_responses.csv"
# female_file = f"../../outputs/socialassumptions/deepseek/{axis}/{axis}_female_1a_deepseek_responses.csv"
# fig_path_net_score = f"../../figuresforpaper/socialassumptions/deepseek/{axis}"

# Make directories if not exist
os.makedirs(fig_path_net_score, exist_ok=True)

# Load data if files exist
dataframes = []
for file in [male_file, female_file]:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dataframes.append(df)
    else:
        print(f"Warning: {file} not found. Skipping.")

# Combine data
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Filter out specific unwanted terms
    unwanted_terms = {"outgoing shy", "social antisocial", "respected disreputable"}
    combined_df = combined_df[~combined_df["term"].isin(unwanted_terms)]

    def to_gender_neutral(descriptor):
        """Convert gendered descriptor to gender-neutral format."""
        descriptor = re.sub(r'\b(male|female|boy|girl)\b', '', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', descriptor, flags=re.IGNORECASE)
        return descriptor

    def adjust_race_ethnicity_descriptor(descriptor, axis_name):
        if axis_name == "race_ethnicity_color" and descriptor == "white american person":
            return "american person"
        return descriptor


    # Apply gender-neutral conversion
    combined_df['descriptor 1'] = combined_df['descriptor 1'].apply(to_gender_neutral)
    combined_df['descriptor 2'] = combined_df['descriptor 2'].apply(to_gender_neutral)
    combined_df['response'] = combined_df['response'].apply(to_gender_neutral)

    # Filter High and Low Valence
    high_df = combined_df[combined_df['direction'] == 'high']
    low_df = combined_df[combined_df['direction'] == 'low']

    # Get unique descriptors dynamically
    unique_descriptors = set(combined_df['descriptor 1'].unique()).union(set(combined_df['descriptor 2'].unique()))

    # Define a fixed order for nationality descriptors
    ability_order = [
        "athletic person", "wheelchair person", "psoriasis person", "healthy person", "ill person",
        "down syndrome person", "disabled person", "glasses person", "blind person", "crippled person"      
    ]
    nationality_order = [
        "african person", "american person", "mexican person", "hispanic person",
        "native american person", "eskimo", "british person", "scottish person", 
        "irish person", "french person", "german person", "greek person", 
        "italian person", "russian person", "indian person", "chinese person", 
        "japanese person", "korean person", "mongolian person", "thai person", 
        "Vietnamese person", "phillipines person", "pakistani person", 
        "iraqi person", "middle eastern person"
    ]

    # Determine x-axis order based on the axis type
    if axis == "nationality":
        xaxis_order = [desc for desc in nationality_order if desc in unique_descriptors]
    elif axis == "ability":
        xaxis_order = [desc for desc in ability_order if desc in unique_descriptors]
    else:
        xaxis_order = sorted(unique_descriptors)

    # Apply race_ethnicity adjustment
    xaxis_order = [adjust_race_ethnicity_descriptor(desc, axis) for desc in xaxis_order]



    # Initialize DataFrame for heatmap
    net_score_df = pd.DataFrame(index=combined_df['taxonomy'].unique(), columns=xaxis_order)

    pvalue_df = pd.DataFrame(index=combined_df['taxonomy'].unique(), columns=xaxis_order)


    # Process each taxonomy
    for taxonomy in combined_df['taxonomy'].unique():
        print(f"Processing taxonomy: {taxonomy}")

        # Filter data for the specific taxonomy
        high_df_taxonomy = high_df[high_df['taxonomy'] == taxonomy]
        low_df_taxonomy = low_df[low_df['taxonomy'] == taxonomy]

        # Count total times each descriptor was an option in High and Low Valence
        descriptor_counts_high = pd.concat([high_df_taxonomy['descriptor 1'], high_df_taxonomy['descriptor 2']]).value_counts()
        descriptor_counts_low = pd.concat([low_df_taxonomy['descriptor 1'], low_df_taxonomy['descriptor 2']]).value_counts()

        # Count times each descriptor was chosen in High and Low Valence
        chosen_counts_high = high_df_taxonomy['response'].value_counts()
        chosen_counts_low = low_df_taxonomy['response'].value_counts()

        # Convert descriptor counts to gender-neutral dynamically
        # descriptor_counts_high.index = descriptor_counts_high.index.map(to_gender_neutral)
        # descriptor_counts_low.index = descriptor_counts_low.index.map(to_gender_neutral)
        descriptor_counts_high.index = descriptor_counts_high.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))
        descriptor_counts_low.index = descriptor_counts_low.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))

        chosen_counts_high.index = chosen_counts_high.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))
        chosen_counts_low.index = chosen_counts_low.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))



        # Compute selection frequency
        high_valence_percentage = (chosen_counts_high / descriptor_counts_high) * 100
        low_valence_percentage = (chosen_counts_low / descriptor_counts_low) * 100

        # Fill missing descriptors with 0
        high_valence_percentage = high_valence_percentage.reindex(xaxis_order, fill_value=0)
        low_valence_percentage = low_valence_percentage.reindex(xaxis_order, fill_value=0)

        # Compute Net Score
        net_score = high_valence_percentage - low_valence_percentage

        # Store in the final heatmap DataFrame
        net_score_df.loc[taxonomy] = net_score


        for descriptor in xaxis_order:
            n1 = descriptor_counts_high.get(descriptor, 0)
            x1 = chosen_counts_high.get(descriptor, 0)
            n2 = descriptor_counts_low.get(descriptor, 0)
            x2 = chosen_counts_low.get(descriptor, 0)
            
            # Build contingency table
            table = np.array([
                [x1, n1 - x1],
                [x2, n2 - x2]
            ])
            
            # Only test if no zeros in both rows
            # if table.min() >= 0 and table.sum() > 0:
            if not np.isnan(table).any() and table.min() >= 0 and table.sum() > 0:
                try:
                    _, p_value = fisher_exact(table)
                except Exception as e:
                    p_value = np.nan
            else:
                p_value = np.nan
            
            pvalue_df.loc[taxonomy, descriptor] = p_value


    # Convert to float (ensuring numeric values)
    net_score_df = net_score_df.astype(float)
    net_score_df = net_score_df.sort_index()



    # ==================== HEATMAP PLOT ==================== #

    def plot_heatmap(data, title, cmap, fig_path, significance_results):
        """Function to plot heatmap."""
        fig, ax = plt.subplots(figsize=(max(16, 0.5 * data.shape[1]), max(6, 0.5 * data.shape[0])))

        # Prepare formatted annotations
        annotations = data.round(2).astype(str)
        for row_idx, row in enumerate(data.index):
            for col_idx, col in enumerate(data.columns):
                p_value = significance_results.loc[row, col]
                if not pd.isna(p_value) and p_value < 0.05:
                    annotations.loc[row, col] = f"$\\bf{{{annotations.loc[row, col]}}}$"
                    # annotations.loc[row, col] = f"{annotations.loc[row, col]} ☀"
                    annotations.loc[row, col] = f"$\\bf{{{annotations.loc[row, col]}}}$ ♥"
                    # annotations.loc[row, col] = f"${annotations.loc[row, col]}^{{\\heartsuit}}$"



        # heatmap = sns.heatmap(
        #     data, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, ax=ax,
        #     annot_kws={"size": 14}, cbar=True, cbar_kws={"pad": 0.02}
        # )
        # heatmap = sns.heatmap(
        #     data, annot=annotations, cmap=cmap, fmt="", linewidths=.5, ax=ax,
        #     annot_kws={"size": 14}, cbar=True, cbar_kws={"pad": 0.02}
        # )
        heatmap = sns.heatmap(
            data, annot=annotations, cmap=cmap, fmt="", 
            linewidths=1.5, linecolor='black',  # <--- add this
            ax=ax, annot_kws={"size": 14}, 
            cbar=True, cbar_kws={"pad": 0.02}
        )
        # Make outer border darker and thicker
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)  # Increase this for thicker outer border

        # Add heart overlay in top-right corner
        for i, row in enumerate(data.index):
            for j, col in enumerate(data.columns):
                p_value = significance_results.loc[row, col]
                if not pd.isna(p_value) and p_value < 0.05:
                    ax.annotate('♥',
                    # ax.annotate('*',
                                xy=(j + 0.80, i + 0.25),  # cell center
                                xytext=(8, 8),         # offset (pixels)
                                textcoords='offset points',
                                color='black',
                                fontsize=10,
                                ha='right',
                                va='top',
                                annotation_clip=False)

        ax.set_title(title, fontsize=16, fontweight='bold')

        # Wrap x-axis labels by inserting "\n" before "person"
        new_labels = [
            label.get_text().replace(" person", "\nperson") if "down syndrome" not in label.get_text() and "college student" not in label.get_text() 
            else label.get_text().replace("down syndrome person", "down\nsyndrome\nperson").replace("college student", "college\nstudent") 
            for label in ax.get_xticklabels()
        ]

        ax.set_xticklabels(new_labels, ha="center")

        ax.tick_params(axis='x', labelsize=16, rotation=90, width=3, pad=5)
        ax.tick_params(axis='y', labelsize=16, rotation=0, width=3)

        heatmap.collections[0].colorbar.ax.tick_params(labelsize=16, width=3)

        # Save the plot
        plt.tight_layout()
        save_path = os.path.join(fig_path_net_score, f"{axis}_net_score_heatmap.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved: {save_path}")

    # Generate the heatmap
    print("Generating heatmap...")
    plot_heatmap(net_score_df, '', blended_cmap, fig_path_net_score, pvalue_df)

# PLOT 2
# ==================== HEATMAP PLOT FOR TERMS (PER TAXONOMY) ==================== #

def plot_term_heatmap(data, title, cmap, save_dir, filename, taxonomy, significance_results):
    height = max(6, 0.5 * data.shape[0])
    if taxonomy == "Politics":
        height = 4  # reduced height for politics
    fig, ax = plt.subplots(figsize=(max(16, 0.5 * data.shape[1]), height), constrained_layout=True)

    # Prepare formatted annotations
    annotations = data.round(2).astype(str)
    for row_idx, row in enumerate(data.index):
        for col_idx, col in enumerate(data.columns):
            p_value = significance_results.loc[row, col]
            if not pd.isna(p_value) and p_value < 0.05:
                # annotations.loc[row, col] = f"$\\bf{{{annotations.loc[row, col]}}}$"
                annotations.loc[row, col] = f"{annotations.loc[row, col]} ☀"



    # heatmap = sns.heatmap(
    #     data, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, ax=ax,
    #     annot_kws={"size": 14}, cbar=True, cbar_kws={"pad": 0.02}
    # )
    # heatmap = sns.heatmap(
    #     data, annot=annotations, cmap=cmap, fmt="", linewidths=.5, ax=ax,
    #     annot_kws={"size": 14}, cbar=True, cbar_kws={"pad": 0.02}
    # )
    heatmap = sns.heatmap(
        data, annot=annotations, cmap=cmap, fmt="", 
        linewidths=1.5, linecolor='black',  # <--- add this
        ax=ax, annot_kws={"size": 14}, 
        cbar=True, cbar_kws={"pad": 0.02}
    )
    # Make outer border darker and thicker
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)  # Increase this for thicker outer border

    ax.set_title(title, fontsize=16, fontweight='bold')

    # ability 12, age 10, nationality 20, race 16, religion 14, socioeconomic 45

    # Wrap x-axis labels by inserting "\n" before "person"
    new_labels = [
        label.get_text().replace(" person", "\nperson") if "down syndrome" not in label.get_text() and "college student" not in label.get_text() 
        else label.get_text().replace("down syndrome person", "down\nsyndrome\nperson").replace("college student", "college\nstudent") 
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels, ha="center")

    new_y_labels = []
    for label in ax.get_yticklabels():
        words = label.get_text().split()
        new_label = "\n".join([" ".join(words[i:i+2]) for i in range(0, len(words), 2)])
        new_y_labels.append(new_label)

    ax.set_yticklabels(new_y_labels, ha="right")

    ax.tick_params(axis='x', labelsize=14, rotation=90, width=2, pad=5)
    ax.tick_params(axis='y', labelsize=14, rotation=0, width=2)

    heatmap.collections[0].colorbar.ax.tick_params(labelsize=14, width=2)

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

# ==================== PLOT 2: TERM-BASED HEATMAP PER TAXONOMY ==================== #

# Load SSCM Dictionary
sscm_dict_path = "../../data/assumptions/sscm_dictionary.csv"
if os.path.exists(sscm_dict_path):
    sscm_df = pd.read_csv(sscm_dict_path)
else:
    raise FileNotFoundError(f"SSCM dictionary not found at {sscm_dict_path}")

# Create a mapping from pair_id to term
pair_id_to_term = dict(zip(sscm_df["pair_id"], sscm_df["term"]))

# Create a mapping of high-low pairs
high_low_pairs = {}
for pair_id in pair_id_to_term.keys():
    if pair_id.startswith("high_"):
        base_id = pair_id.replace("high_", "")
        low_id = f"low_{base_id}"
        if low_id in pair_id_to_term:
            high_low_pairs[pair_id] = low_id

# Create directory to save taxonomy-wise term heatmaps
term_taxonomy_save_dir = os.path.join(fig_path_net_score, "term_taxonomy_heatmaps")
os.makedirs(term_taxonomy_save_dir, exist_ok=True)

# Process each taxonomy separately
for taxonomy in combined_df["taxonomy"].unique():
    print(f"Generating term-based heatmaps for taxonomy: {taxonomy}")

    # Filter data for the specific taxonomy
    taxonomy_df = combined_df[combined_df["taxonomy"] == taxonomy]

    # Get terms specific to this taxonomy
    taxonomy_terms = set(taxonomy_df["term"].unique())

    # Filter only the term pairs relevant to this taxonomy
    filtered_high_low_pairs = {
        high: low for high, low in high_low_pairs.items()
        if pair_id_to_term[high] in taxonomy_terms and pair_id_to_term[low] in taxonomy_terms
    }

    if not filtered_high_low_pairs:
        print(f"Skipping {taxonomy} as no matching term pairs found.")
        continue  # Skip this taxonomy if no matching pairs

    # Initialize DataFrame for heatmap
    net_score_term_taxonomy_df = pd.DataFrame(
        index=[f"{pair_id_to_term[high]} - {pair_id_to_term[low]}" for high, low in filtered_high_low_pairs.items()],
        columns=xaxis_order
    )

    # Process each term pair for the specific taxonomy
    for high, low in filtered_high_low_pairs.items():
        high_df_term = taxonomy_df[taxonomy_df['term'] == pair_id_to_term[high]]
        low_df_term = taxonomy_df[taxonomy_df['term'] == pair_id_to_term[low]]

        descriptor_counts_high = pd.concat([high_df_term['descriptor 1'], high_df_term['descriptor 2']]).value_counts()
        descriptor_counts_low = pd.concat([low_df_term['descriptor 1'], low_df_term['descriptor 2']]).value_counts()

        chosen_counts_high = high_df_term['response'].value_counts()
        chosen_counts_low = low_df_term['response'].value_counts()

        # Convert descriptor counts to gender-neutral dynamically
        # descriptor_counts_high.index = descriptor_counts_high.index.map(to_gender_neutral)
        # descriptor_counts_low.index = descriptor_counts_low.index.map(to_gender_neutral)
        descriptor_counts_high.index = descriptor_counts_high.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))
        descriptor_counts_low.index = descriptor_counts_low.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))

        chosen_counts_high.index = chosen_counts_high.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))
        chosen_counts_low.index = chosen_counts_low.index.map(to_gender_neutral).map(lambda d: adjust_race_ethnicity_descriptor(d, axis))



        # Compute selection frequency
        high_valence_percentage = (chosen_counts_high / descriptor_counts_high) * 100
        low_valence_percentage = (chosen_counts_low / descriptor_counts_low) * 100

        # Fill missing descriptors with 0
        high_valence_percentage = high_valence_percentage.reindex(xaxis_order, fill_value=0)
        low_valence_percentage = low_valence_percentage.reindex(xaxis_order, fill_value=0)

        # Compute Net Score
        net_score = high_valence_percentage - low_valence_percentage

        # Ensure missing values are treated as zero
        net_score = net_score.fillna(0)

        # Store in the final heatmap DataFrame
        net_score_term_taxonomy_df.loc[f"{pair_id_to_term[high]} - {pair_id_to_term[low]}"] = net_score

        # Reindex to ensure all descriptors exist
        net_score_term_taxonomy_df = net_score_term_taxonomy_df.reindex(columns=xaxis_order, fill_value=0)


    # Convert to float and sort
    net_score_term_taxonomy_df = net_score_term_taxonomy_df.astype(float).sort_index()

    # Generate heatmap for the specific taxonomy
    if not net_score_term_taxonomy_df.empty:
        filename = f"{taxonomy}_term_net_score_heatmap.png"
        empty_pvalue_df = pd.DataFrame(index=net_score_term_taxonomy_df.index, columns=net_score_term_taxonomy_df.columns)
        # plot_term_heatmap(net_score_term_taxonomy_df, f"Net Score for {taxonomy}", blended_cmap, term_taxonomy_save_dir, filename, taxonomy, pvalue_df)
        plot_term_heatmap(net_score_term_taxonomy_df, f"Net Score for {taxonomy}", blended_cmap, term_taxonomy_save_dir, filename, taxonomy, empty_pvalue_df)




# ==================== HEATMAP PLOT FOR TERMS (SUBPLOTS) ==================== #

# def plot_term_subplots(taxonomy_heatmaps, save_dir, filename):
#     """Function to plot all taxonomy term-based heatmaps as subplots with a shared x-axis."""
#     num_taxonomies = len(taxonomy_heatmaps)
#     if num_taxonomies == 0:
#         print("No valid taxonomy heatmaps to plot.")
#         return

#     fig, axes = plt.subplots(nrows=num_taxonomies, ncols=1, figsize=(15, 6 * num_taxonomies), sharex=True)

#     if num_taxonomies == 1:
#         axes = [axes]  # Ensure axes is iterable when there's only one taxonomy

#     for i, (ax, (taxonomy, data)) in enumerate(zip(axes, taxonomy_heatmaps.items())):
#         sns.heatmap(
#             data, annot=True, cmap=blended_cmap, fmt=".2f", linewidths=.5, ax=ax,
#             annot_kws={"size": 14}, cbar=True, cbar_kws={"pad": 0.02})
#         ax.set_title(f"{taxonomy}", fontsize=16, fontweight='bold')

#         # Format y-axis labels: Insert "\n" every two words
#         new_y_labels = [
#             "\n".join([" ".join(label.get_text().split()[i:i+2]) for i in range(0, len(label.get_text().split()), 2)])
#             for label in ax.get_yticklabels()
#         ]
#         ax.set_yticklabels(new_y_labels, ha="right")

#         ax.tick_params(axis='y', labelsize=14, rotation=0, width=2, pad=5)

#         # Hide x-axis labels for all but the last plot
#         if i != num_taxonomies - 1:
#             ax.set_xticklabels([])
#             ax.set_xlabel("")
#         else:
#             # Format x-axis labels for better readability (split words)
#             new_x_labels = [
#                 "\n".join([" ".join(label.get_text().split()[i:i+2]) for i in range(0, len(label.get_text().split()), 2)])
#                 for label in ax.get_xticklabels()
#             ]
#             ax.set_xticklabels(new_x_labels, ha="center", fontsize=14, rotation=90)

#             ax.tick_params(axis='x', labelsize=14, rotation=90, width=2, pad=5)

#     plt.tight_layout()
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
#     plt.show()

#     print(f"Saved: {save_path}")


# # ==================== PLOT 2: TERM-BASED HEATMAP SUBPLOTS ==================== #

# # Load SSCM Dictionary
# sscm_dict_path = "../../data/assumptions/sscm_dictionary.csv"
# if os.path.exists(sscm_dict_path):
#     sscm_df = pd.read_csv(sscm_dict_path)
# else:
#     raise FileNotFoundError(f"SSCM dictionary not found at {sscm_dict_path}")

# # Create a mapping from pair_id to term
# pair_id_to_term = dict(zip(sscm_df["pair_id"], sscm_df["term"]))

# # Create a mapping of high-low pairs
# high_low_pairs = {}
# for pair_id in pair_id_to_term.keys():
#     if pair_id.startswith("high_"):
#         base_id = pair_id.replace("high_", "")
#         low_id = f"low_{base_id}"
#         if low_id in pair_id_to_term:
#             high_low_pairs[pair_id] = low_id

# # Dictionary to store heatmaps for all taxonomies
# taxonomy_heatmaps = {}

# # Process each taxonomy separately
# for taxonomy in combined_df["taxonomy"].unique():
#     print(f"Generating term-based heatmaps for taxonomy: {taxonomy}")

#     # Filter data for the specific taxonomy
#     taxonomy_df = combined_df[combined_df["taxonomy"] == taxonomy]

#     # Get terms specific to this taxonomy
#     taxonomy_terms = set(taxonomy_df["term"].unique())

#     # Filter only the term pairs relevant to this taxonomy
#     filtered_high_low_pairs = {
#         high: low for high, low in high_low_pairs.items()
#         if pair_id_to_term[high] in taxonomy_terms and pair_id_to_term[low] in taxonomy_terms
#     }

#     if not filtered_high_low_pairs:
#         print(f"Skipping {taxonomy} as no matching term pairs found.")
#         continue  # Skip this taxonomy if no matching pairs

#     # Initialize DataFrame for heatmap
#     net_score_term_taxonomy_df = pd.DataFrame(
#         index=[f"{pair_id_to_term[high]} - {pair_id_to_term[low]}" for high, low in filtered_high_low_pairs.items()],
#         columns=xaxis_order
#     )

#     # Process each term pair for the specific taxonomy
#     for high, low in filtered_high_low_pairs.items():
#         high_df_term = taxonomy_df[taxonomy_df['term'] == pair_id_to_term[high]]
#         low_df_term = taxonomy_df[taxonomy_df['term'] == pair_id_to_term[low]]

#         descriptor_counts_high = pd.concat([high_df_term['descriptor 1'], high_df_term['descriptor 2']]).value_counts()
#         descriptor_counts_low = pd.concat([low_df_term['descriptor 1'], low_df_term['descriptor 2']]).value_counts()

#         chosen_counts_high = high_df_term['response'].value_counts()
#         chosen_counts_low = low_df_term['response'].value_counts()

#         # Convert descriptor counts to gender-neutral dynamically
#         descriptor_counts_high.index = descriptor_counts_high.index.map(to_gender_neutral)
#         descriptor_counts_low.index = descriptor_counts_low.index.map(to_gender_neutral)

#         # Compute selection frequency
#         high_valence_percentage = (chosen_counts_high / descriptor_counts_high) * 100
#         low_valence_percentage = (chosen_counts_low / descriptor_counts_low) * 100

#         # Fill missing descriptors with 0
#         high_valence_percentage = high_valence_percentage.reindex(xaxis_order, fill_value=0)
#         low_valence_percentage = low_valence_percentage.reindex(xaxis_order, fill_value=0)

#         # Compute Net Score
#         net_score = high_valence_percentage - low_valence_percentage

#         # Ensure missing values are treated as zero
#         net_score = net_score.fillna(0)

#         # Store in the final heatmap DataFrame
#         net_score_term_taxonomy_df.loc[f"{pair_id_to_term[high]} - {pair_id_to_term[low]}"] = net_score

#     # Convert to float and sort
#     net_score_term_taxonomy_df = net_score_term_taxonomy_df.astype(float).sort_index()

#     # Store the heatmap for later plotting
#     taxonomy_heatmaps[taxonomy] = net_score_term_taxonomy_df

# # Generate all heatmaps as subplots
# plot_term_subplots(taxonomy_heatmaps, fig_path_net_score, f"{axis}_all_taxonomy_term_net_score_heatmaps.png")
