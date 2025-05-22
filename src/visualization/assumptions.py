import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import re

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

# Reverse the blended colormap
blended_colors_r = blended_colors[::-1]

# Create new colormap
blended_cmap_r = ListedColormap(blended_colors_r)


# Load colormaps
# Define your custom hex colors
colors = ["#C3B1E1", "#008080"]

# Create the custom colormap
paired_cmap = LinearSegmentedColormap.from_list("PurpleTeal", colors, N=256)

# Optional: reverse the blended colormap
paired_cmap_r = ListedColormap(blended_colors[::-1])

# bias_axes = [
#     "ability", "age", "nationality", "race_ethnicity_color",
#     "physical_traits", "religion", "socioeconomic", "gender_and_sex"
# ]

# Define the axis
axis = "socioeconomic"  # Change this to the desired axis

# File paths
male_file = f"../../../backupresults/outputs/assumptions/flux/{axis}/setting1_2a_{axis}_male_llava_responses.csv"
female_file = f"../../../backupresults/outputs/assumptions/flux/{axis}/setting1_2a_{axis}_female_llava_responses.csv"
fig_path = f"../../figuresforpaper/assumptions/flux/{axis}"

# Make directories if not exist
os.makedirs(fig_path, exist_ok=True)

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

    def to_gender_neutral(descriptor):
        """Convert gendered descriptor to gender-neutral format."""
        descriptor = re.sub(r'\b(male|female|boy|girl)\b', '', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', descriptor, flags=re.IGNORECASE)
        return descriptor.strip()

    # Apply gender-neutral conversion
    combined_df['descriptor 1'] = combined_df['descriptor 1'].apply(to_gender_neutral)
    combined_df['descriptor 2'] = combined_df['descriptor 2'].apply(to_gender_neutral)
    combined_df['response'] = combined_df['response'].apply(to_gender_neutral)

    # Create a concatenated descriptor column
    combined_df['descriptors'] = combined_df['descriptor 1'] + ' & ' + combined_df['descriptor 2']

    # Calculate response occurrences per activity
    response_counts = combined_df.groupby(['activity', 'response']).size().reset_index(name='response_count')
    option_counts = combined_df.melt(id_vars=['activity', 'response'], 
                                     value_vars=['descriptor 1', 'descriptor 2'],
                                     var_name='descriptor_type',
                                     value_name='option').groupby(['activity', 'option']).size().reset_index(name='option_count')

    # Merge to calculate percentage
    result_df = response_counts.merge(option_counts, left_on=['activity', 'response'], right_on=['activity', 'option'])
    result_df['percentage'] = (result_df['response_count'] / result_df['option_count']) * 100

    # Calculate p-values for percentage heatmap
    heatmap_data = result_df.pivot(index='response', columns='activity', values='percentage').fillna(0)

    pvalue_percentage_df = pd.DataFrame(index=heatmap_data.index, columns=heatmap_data.columns)
    total_response = response_counts.groupby('response')['response_count'].sum()
    total_option = option_counts.groupby('option')['option_count'].sum()

    for idx, row in result_df.iterrows():
        A, R = row['activity'], row['response']
        r_A = row['response_count']
        o_A = row['option_count']
        r_negA = total_response[R] - r_A
        o_negA = total_option[R] - o_A

        table = np.array([[r_A, o_A - r_A], [r_negA, o_negA - r_negA]])
        if not np.isnan(table).any() and table.min() >= 0 and table.sum() > 0:
            try:
                _, p_value = fisher_exact(table)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        pvalue_percentage_df.loc[R, A] = p_value

    pvalue_percentage_df = pvalue_percentage_df.astype(float)


    # Pivot for heatmap
    # heatmap_data = result_df.pivot(index='response', columns='activity', values='percentage').fillna(0)

    # Plot heatmap
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(max(50, len(heatmap_data.columns) * 0.5), max(6, len(heatmap_data.index) * 0.3)))
    annotations = heatmap_data.round(1).astype(str)
    for row in heatmap_data.index:
        for col in heatmap_data.columns:
            pval = pvalue_percentage_df.loc[row, col]
            if not pd.isna(pval) and pval < 0.05:
                annotations.loc[row, col] = f"$\\bf{{{annotations.loc[row, col]}}}$"

    plt.figure(figsize=(max(50, len(heatmap_data.columns) * 0.5), max(6, len(heatmap_data.index) * 0.3)))
    ax = sns.heatmap(heatmap_data, cmap=blended_cmap, cbar=True, cbar_kws={"pad": 0.02},
                    annot=annotations, fmt="", linewidths=0.5)

    for i, row in enumerate(heatmap_data.index):
        for j, col in enumerate(heatmap_data.columns):
            pval = pvalue_percentage_df.loc[row, col]
            if not pd.isna(pval) and pval < 0.05:
                ax.annotate('♥', xy=(j + 0.80, i + 0.25), xytext=(8, 8),
                            textcoords='offset points', color='black',
                            fontsize=10, ha='right', va='top', annotation_clip=False)

    plt.xlabel("Activity")
    plt.ylabel("Descriptor")
    plt.title("Response Percentage by Activity and Descriptor")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"setting1_2a_{axis}_response_vs_activity_heatmap.png"), format="png", dpi=300, bbox_inches="tight")
    # plt.savefig(heatmap_output_file, format="png", dpi=300, bbox_inches="tight")
    plt.show()


def calculate_log_odds(response_counts, option_counts, smoothing=1):
    """
    Calculate log-odds per (activity, response) relative to all other activities.
    """
    # Merge response and option counts
    merged = response_counts.merge(option_counts, left_on=['activity', 'response'], right_on=['activity', 'option'])
    merged = merged.rename(columns={'response_count': 'r', 'option_count': 'o'})

    # Compute total counts across all activities (for neg A)
    total_response = response_counts.groupby('response')['response_count'].sum().reset_index(name='total_r')
    total_option = option_counts.groupby('option')['option_count'].sum().reset_index(name='total_o')

    # Merge in totals
    merged = merged.merge(total_response, left_on='response', right_on='response')
    total_option = total_option.rename(columns={'option': 'response'})
    merged = merged.merge(total_option, on='response')


    # Calculate counts outside A (neg A)
    merged['r_negA'] = merged['total_r'] - merged['r']
    merged['o_negA'] = merged['total_o'] - merged['o']

    # Calculate odds with smoothing
    merged['odds_A'] = (merged['r'] + smoothing) / ((merged['o'] - merged['r']).clip(lower=0) + smoothing)
    merged['odds_negA'] = (merged['r_negA'] + smoothing) / ((merged['o_negA'] - merged['r_negA']).clip(lower=0) + smoothing)


    # Compute log-odds
    merged['log_odds'] = np.log(merged['odds_A'] / merged['odds_negA'])

    return merged[['activity', 'response', 'log_odds']].rename(columns={'response': 'descriptor'})


def calculate_log_odds_pvalues(log_odds_df, response_counts, option_counts):
    pval_df = pd.DataFrame(index=log_odds_df['descriptor'].unique(), columns=log_odds_df['activity'].unique())
    total_response = response_counts.groupby('response')['response_count'].sum()
    total_option = option_counts.groupby('option')['option_count'].sum()

    for _, row in log_odds_df.iterrows():
        A, R = row['activity'], row['descriptor']
        r_A = response_counts.query("activity == @A and response == @R")['response_count'].sum()
        o_A = option_counts.query("activity == @A and option == @R")['option_count'].sum()
        r_negA = total_response[R] - r_A
        o_negA = total_option[R] - o_A

        table = np.array([[r_A, o_A - r_A], [r_negA, o_negA - r_negA]])
        if not np.isnan(table).any() and table.min() >= 0 and table.sum() > 0:
            try:
                _, p_value = fisher_exact(table)
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        pval_df.loc[R, A] = p_value

    return pval_df.astype(float)



# Calculate log-odds
log_odds_df = calculate_log_odds(response_counts, option_counts)

pvalue_logodds_df = calculate_log_odds_pvalues(log_odds_df, response_counts, option_counts)


# Pivot for heatmap (optional)
log_odds_heatmap = log_odds_df.pivot(index='descriptor', columns='activity', values='log_odds').fillna(0)

# Plot log-odds heatmap
plt.figure(figsize=(max(50, len(log_odds_heatmap.columns) * 0.5), max(6, len(log_odds_heatmap.index) * 0.3)))
annotations_logodds = log_odds_heatmap.round(2).astype(str)
for row in log_odds_heatmap.index:
    for col in log_odds_heatmap.columns:
        pval = pvalue_logodds_df.loc[row, col]
        if not pd.isna(pval) and pval < 0.05:
            annotations_logodds.loc[row, col] = f"$\\bf{{{annotations_logodds.loc[row, col]}}}$"

plt.figure(figsize=(max(50, len(log_odds_heatmap.columns) * 0.5), max(6, len(log_odds_heatmap.index) * 0.3)))
ax = sns.heatmap(log_odds_heatmap, cmap=blended_cmap, cbar=True, cbar_kws={"pad": 0.02},
                 annot=annotations_logodds, fmt="", linewidths=0.5)

for i, row in enumerate(log_odds_heatmap.index):
    for j, col in enumerate(log_odds_heatmap.columns):
        pval = pvalue_logodds_df.loc[row, col]
        if not pd.isna(pval) and pval < 0.05:
            ax.annotate('♥', xy=(j + 0.80, i + 0.25), xytext=(8, 8),
                        textcoords='offset points', color='black',
                        fontsize=10, ha='right', va='top', annotation_clip=False)

plt.xlabel("Activity")
plt.ylabel("Descriptor")
plt.title("Log-Odds by Activity and Descriptor")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"setting1_2a_{axis}_log_odds_heatmap.png"), format="png", dpi=300, bbox_inches="tight")
plt.show()

# x vs y

def calculate_pairwise_competition_and_significance(df):
    descriptors = pd.unique(df[['descriptor 1', 'descriptor 2']].values.ravel())
    result_matrix = pd.DataFrame(index=descriptors, columns=descriptors, dtype=float)
    pvalue_matrix = pd.DataFrame(index=descriptors, columns=descriptors, dtype=float)

    for X in descriptors:
        for Y in descriptors:
            if X == Y:
                result_matrix.loc[X, Y] = np.nan
                pvalue_matrix.loc[X, Y] = np.nan
                continue

            # Rows with X and Y together
            with_Y = df[
                ((df['descriptor 1'] == X) & (df['descriptor 2'] == Y)) |
                ((df['descriptor 1'] == Y) & (df['descriptor 2'] == X))
            ]
            n_with_Y = len(with_Y)
            n_X_selected_with_Y = (with_Y['response'] == X).sum()

            # Rows with X but no Y
            without_Y = df[
                ((df['descriptor 1'] == X) & (df['descriptor 2'] != Y)) |
                ((df['descriptor 2'] == X) & (df['descriptor 1'] != Y))
            ]
            n_without_Y = len(without_Y)
            n_X_selected_without_Y = (without_Y['response'] == X).sum()

            # Calculate proportions
            if n_with_Y > 0:
                p_X_with_Y = n_X_selected_with_Y / n_with_Y
            else:
                p_X_with_Y = np.nan

            if n_without_Y > 0:
                p_X_without_Y = n_X_selected_without_Y / n_without_Y
            else:
                p_X_without_Y = np.nan

            # Store difference
            diff = p_X_with_Y - p_X_without_Y
            result_matrix.loc[X, Y] = diff

            # Build contingency table
            table = np.array([
                [n_X_selected_with_Y, n_with_Y - n_X_selected_with_Y],
                [n_X_selected_without_Y, n_without_Y - n_X_selected_without_Y]
            ])
            if np.all(table.sum(axis=1) > 0) and np.all(table.sum(axis=0) > 0):
                _, p_value = fisher_exact(table)
            else:
                p_value = np.nan

            pvalue_matrix.loc[X, Y] = p_value

    return result_matrix, pvalue_matrix


def plot_pairwise_heatmap(data, pvalues, title, cmap='PRBG'):
    annotations = data.round(2).astype(str)
    for i in data.index:
        for j in data.columns:
            pval = pvalues.loc[i, j]
            if pd.notna(pval) and pval < 0.05:
                annotations.loc[i, j] = f"$\\bf{{{annotations.loc[i, j]}}}$"

#   ability 14, age 14, nationality 30, race 20, religion 16, physical 40, socio 50, gender 8
    plt.figure(figsize=(max(50, 0.4 * len(data.columns)), max(50, 0.4 * len(data.index))))
    # plt.figure(figsize=(1 * len(data.columns), 1 * len(data.index)))
    ax = sns.heatmap(data, annot=annotations, fmt="", cmap=cmap,
                     linewidths=1, linecolor='black', cbar_kws={"pad": 0.02}, annot_kws={"size": 16})
    
    for i, row in enumerate(data.index):
        for j, col in enumerate(data.columns):
            pval = pvalues.loc[row, col]
            if pd.notna(pval) and pval < 0.05:
                ax.annotate('♥', xy=(j + 0.7, i + 0.2), xytext=(8, 8),
                            textcoords='offset points', color='black',
                            fontsize=16, ha='center', va='top', annotation_clip=False)

    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=20, fontweight='medium')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20, fontweight='medium')
    ax.tick_params(axis='x', width=2, pad=5)
    ax.tick_params(axis='y', width=2)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20, width=3)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)


    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"setting1_2a_{axis}_pairwise_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()


pairwise_diff_file = os.path.join(fig_path, 'setting1_2a_pairwise_diff.csv')
pairwise_pvals_file = os.path.join(fig_path, 'setting1_2a_pairwise_pvals.csv')

if os.path.exists(pairwise_diff_file) and os.path.exists(pairwise_pvals_file):
    pairwise_diff = pd.read_csv(pairwise_diff_file, index_col=0)
    pairwise_pvals = pd.read_csv(pairwise_pvals_file, index_col=0)
    print("✅ Loaded existing pairwise CSV files.")
else:
    pairwise_diff, pairwise_pvals = calculate_pairwise_competition_and_significance(combined_df)
    pairwise_diff = pairwise_diff.sort_index().sort_index(axis=1)
    pairwise_pvals = pairwise_pvals.loc[pairwise_diff.index, pairwise_diff.columns]
    np.fill_diagonal(pairwise_diff.values, 0)
    pairwise_diff.to_csv(pairwise_diff_file)
    pairwise_pvals.to_csv(pairwise_pvals_file)
    print("✅ Calculated and saved new pairwise CSV files.")


pairwise_diff.index = pairwise_diff.index.str.replace(' person', '', regex=False)
pairwise_diff.columns = pairwise_diff.columns.str.replace(' person', '', regex=False)
pairwise_pvals.index = pairwise_pvals.index.str.replace(' person', '', regex=False)
pairwise_pvals.columns = pairwise_pvals.columns.str.replace(' person', '', regex=False)

plot_pairwise_heatmap(pairwise_diff, pairwise_pvals, 
                      title='Pairwise Competition Metric (ΔX,Y)', cmap=paired_cmap)