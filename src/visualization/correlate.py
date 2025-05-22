import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.lines as mlines

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

axis="race_ethnicity_color"

# df = pd.read_csv("../../outputs/factuality/deepseek/setting1_ability_male_deepseek_responses.csv")
# print(df.columns.tolist())

def compute_factuality_scores(input_dir, setting=1, save_path=None):
    """
    Compute factuality score per descriptor — i.e., for each descriptor, 
    compute the % of times it was involved in a pair and the model chose the correct activity.
    """
    import os
    import pandas as pd
    import numpy as np
    import re
    from scipy.stats import fisher_exact

    input_files = [
        os.path.join(input_dir, f"setting{setting}_{axis}_male_deepseek_responses.csv"),
        os.path.join(input_dir, f"setting{setting}_{axis}_female_deepseek_responses.csv")
    ]

    combined = []
    for file in input_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            combined.append(df)
        else:
            print(f"⚠️ Warning: {file} not found. Skipping.")

    if not combined:
        return pd.DataFrame()

    df = pd.concat(combined, ignore_index=True)

    def to_gender_neutral(desc):
        desc = str(desc)
        desc = re.sub(r'\b(male|female)\b', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\b(man|woman|boy|girl)\b', 'person', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', desc, flags=re.IGNORECASE)
        return desc.strip()

    df["descriptor 1"] = df["descriptor 1"].apply(to_gender_neutral)
    df["descriptor 2"] = df["descriptor 2"].apply(to_gender_neutral)
    df["response"] = df["response"].astype(str).apply(to_gender_neutral)
    df["activity"] = df["activity"].str.strip()

    # Compare activity (ground truth) to response
    df["is_correct"] = df["activity"] == df["response"]

    # For each row, assign both descriptors, mark which was correct
    descriptor_records = []

    for _, row in df.iterrows():
        d1 = row["descriptor 1"]
        d2 = row["descriptor 2"]
        correct = row["is_correct"]

        for desc in [d1, d2]:
            descriptor_records.append({
                "Descriptor": desc,
                "Correct": int(correct)
            })

    desc_df = pd.DataFrame(descriptor_records)
    grouped = desc_df.groupby("Descriptor")["Correct"].agg(["sum", "count"]).reset_index()
    grouped.rename(columns={"sum": "Correct", "count": "Total"}, inplace=True)
    grouped["Factuality Score"] = 100 * grouped["Correct"] / grouped["Total"]

    # Fisher's test
    total_correct = grouped["Correct"].sum()
    total_total = grouped["Total"].sum()
    pvals = []
    for _, row in grouped.iterrows():
        x1 = row["Correct"]
        n1 = row["Total"]
        x2 = total_correct - x1
        n2 = total_total - n1
        table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
        if table.min() >= 0 and table.sum() > 0:
            try:
                _, p = fisher_exact(table)
            except:
                p = np.nan
        else:
            p = np.nan
        pvals.append(p)

    grouped["p_value"] = pvals

    result_df = grouped[["Descriptor", "Factuality Score", "p_value"]].sort_values("Factuality Score", ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False)
        print(f"✅ Factuality scores saved to {save_path}")

    return result_df


def compute_perception_scores(male_file, female_file, save_path=None):

    def to_gender_neutral(descriptor):
        descriptor = re.sub(r'\b(male|female|boy|girl)\b', '', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', descriptor, flags=re.IGNORECASE)
        return descriptor.strip()

    # Load both files
    dfs = []
    for file in [male_file, female_file]:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"⚠️ Warning: {file} not found. Skipping.")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Normalize descriptors
    df['descriptor 1'] = df['descriptor 1'].apply(to_gender_neutral)
    df['descriptor 2'] = df['descriptor 2'].apply(to_gender_neutral)
    df['response'] = df['response'].apply(to_gender_neutral)

    # Count how often each descriptor was shown (in options) and chosen (in response)
    all_descriptors = pd.concat([df['descriptor 1'], df['descriptor 2']])
    option_counts = all_descriptors.value_counts().to_dict()
    response_counts = df['response'].value_counts().to_dict()

    # Aggregate perception percentnationality and compute p-values
    records = []
    total_chosen = sum(response_counts.values())
    total_shown = sum(option_counts.values())

    for desc in option_counts:
        shown = option_counts.get(desc, 0)
        chosen = response_counts.get(desc, 0)
        perc = (chosen / shown) * 100 if shown > 0 else 0

        # Fisher exact test
        x1 = chosen
        n1 = shown
        x2 = total_chosen - x1
        n2 = total_shown - n1
        table = np.array([[x1, n1 - x1], [x2, n2 - x2]])

        if table.min() >= 0 and table.sum() > 0:
            try:
                _, p_value = fisher_exact(table)
            except:
                p_value = np.nan
        else:
            p_value = np.nan

        records.append({
            "Descriptor": desc,
            "Perception Score": perc,  # LOWER means worse perception
            "p_value": p_value
        })

    result_df = pd.DataFrame(records).sort_values("Perception Score", ascending=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False)
        print(f"✅ Perception scores saved to {save_path}")

    return result_df

def compute_stereotype_scores(male_file, female_file, save_path=None):
    """
    Compute stereotype net score per descriptor by averaging across taxonomy rows.
    Positive score = descriptor is chosen more often in high-valence contexts.
    """

    def to_gender_neutral(descriptor):
        descriptor = re.sub(r'\b(male|female|boy|girl)\b', '', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(man|woman|boy|girl)\b', 'person', descriptor, flags=re.IGNORECASE)
        descriptor = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', descriptor, flags=re.IGNORECASE)
        return descriptor.strip()

    def adjust_race_ethnicity_descriptor(descriptor):
        if axis == "nationality" and descriptor == "white american person":
            return "american person"
        return descriptor

    # Load both files
    dfs = []
    for file in [male_file, female_file]:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"⚠️ Warning: {file} not found. Skipping.")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df[~df["term"].isin({"outgoing shy", "social antisocial", "respected disreputable"})]

    # Normalize
    df['descriptor 1'] = df['descriptor 1'].apply(to_gender_neutral).apply(adjust_race_ethnicity_descriptor)
    df['descriptor 2'] = df['descriptor 2'].apply(to_gender_neutral).apply(adjust_race_ethnicity_descriptor)
    df['response']     = df['response'].apply(to_gender_neutral).apply(adjust_race_ethnicity_descriptor)

    # Separate high/low valence
    high_df = df[df["direction"] == "high"]
    low_df  = df[df["direction"] == "low"]

    all_taxonomies = df['taxonomy'].unique()
    all_descriptors = sorted(set(df['descriptor 1']).union(df['descriptor 2']))

    net_scores = {desc: [] for desc in all_descriptors}
    p_values = {desc: [] for desc in all_descriptors}

    # For each taxonomy
    for taxonomy in all_taxonomies:
        high = high_df[high_df["taxonomy"] == taxonomy]
        low  = low_df[low_df["taxonomy"] == taxonomy]

        # Count total appearances and selections
        opt_count_high = pd.concat([high["descriptor 1"], high["descriptor 2"]]).value_counts()
        opt_count_low  = pd.concat([low["descriptor 1"], low["descriptor 2"]]).value_counts()

        resp_count_high = high["response"].value_counts()
        resp_count_low  = low["response"].value_counts()

        for desc in all_descriptors:
            x1 = resp_count_high.get(desc, 0)
            n1 = opt_count_high.get(desc, 0)
            x2 = resp_count_low.get(desc, 0)
            n2 = opt_count_low.get(desc, 0)

            p_high = (x1 / n1) * 100 if n1 > 0 else 0
            p_low  = (x2 / n2) * 100 if n2 > 0 else 0

            net = p_high - p_low
            net_scores[desc].append(net)

            # Fisher exact test
            table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
            if table.min() >= 0 and table.sum() > 0:
                try:
                    _, p_val = fisher_exact(table)
                except:
                    p_val = np.nan
            else:
                p_val = np.nan
            p_values[desc].append(p_val)

    # Aggregate: avernationality across taxonomies
    result_records = []
    for desc in all_descriptors:
        scores = net_scores[desc]
        pvals = [p for p in p_values[desc] if not pd.isna(p)]
        result_records.append({
            "Descriptor": desc,
            "Stereotype Score": np.mean(scores),
            "p_value": min(pvals) if pvals else np.nan
        })

    result_df = pd.DataFrame(result_records).sort_values("Stereotype Score", ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False)
        print(f"✅ Stereotype scores saved to {save_path}")

    return result_df


def compute_decisionmaking_scores(male_file, female_file, save_path=None):
    """
    Compute decision-making scores per descriptor:
    Percentnationality of times each descriptor was selected when presented as an option.
    """

    def to_gender_neutral(desc):
        desc = str(desc)
        desc = re.sub(r'\b(male|female)\b', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\b(man|woman|boy|girl)\b', 'person', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\b(filipino|filipina)\b', 'phillipines person', desc, flags=re.IGNORECASE)
        return desc.strip()

    # Load files
    dfs = []
    for file in [male_file, female_file]:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"⚠️ Warning: {file} not found. Skipping.")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["descriptor 1"] = df["descriptor 1"].apply(to_gender_neutral)
    df["descriptor 2"] = df["descriptor 2"].apply(to_gender_neutral)
    df["response"] = df["response"].apply(to_gender_neutral)

    # Count options and selections
    descriptor_counts = pd.concat([df["descriptor 1"], df["descriptor 2"]]).value_counts()
    response_counts = df["response"].value_counts()

    records = []
    total_responses = response_counts.sum()
    total_options = descriptor_counts.sum()

    for desc in descriptor_counts.index:
        x1 = response_counts.get(desc, 0)
        n1 = descriptor_counts[desc]
        x2 = total_responses - x1
        n2 = total_options - n1

        score = (x1 / n1) * 100 if n1 > 0 else 0

        table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
        if table.min() >= 0 and table.sum() > 0:
            try:
                _, p = fisher_exact(table)
            except:
                p = np.nan
        else:
            p = np.nan

        records.append({
            "Descriptor": desc,
            "Decisionmaking Score": score,
            "p_value": p
        })

    result_df = pd.DataFrame(records).sort_values("Decisionmaking Score", ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False)
        print(f"✅ Decisionmaking scores saved to {save_path}")

    return result_df

def merge_task_scores(axis, base_dir="intermediate", save=True):
    """
    Load, align, and normalize factuality, perception, stereotype, and decisionmaking scores.
    Ensures all scores are in the same direction: higher = better.
    Normalizes stereotype to 0–100 range and inverts perception scores.
    """

    # File paths
    factuality_file = os.path.join(base_dir, f"{axis}_factuality_scores.csv")
    perception_file = os.path.join(base_dir, f"{axis}_perception_scores.csv")
    stereotype_file = os.path.join(base_dir, f"{axis}_stereotype_scores.csv")
    decision_file = os.path.join(base_dir, f"{axis}_decisionmaking_scores.csv")

    # Load
    factuality = pd.read_csv(factuality_file)[["Descriptor", "Factuality Score"]]
    perception = pd.read_csv(perception_file)[["Descriptor", "Perception Score"]]
    stereotype = pd.read_csv(stereotype_file)[["Descriptor", "Stereotype Score"]]
    decision = pd.read_csv(decision_file)[["Descriptor", "Decisionmaking Score"]]

    # Merge on descriptor
    merged = factuality.merge(perception, on="Descriptor", how="outer")\
                       .merge(stereotype, on="Descriptor", how="outer")\
                       .merge(decision, on="Descriptor", how="outer")

    # Invert perception (lower = worse → higher = better)
    if "Perception Score" in merged.columns:
        merged["Perception Score"] = 100 - merged["Perception Score"]

    # Normalize Stereotype Score (min-max to 0–100)
    if "Stereotype Score" in merged.columns:
        min_s = merged["Stereotype Score"].min()
        max_s = merged["Stereotype Score"].max()
        if max_s != min_s:
            merged["Stereotype Score"] = 100 * (merged["Stereotype Score"] - min_s) / (max_s - min_s)
        else:
            merged["Stereotype Score"] = 50  # constant neutral if all scores are same

    # Drop descriptors missing all scores (very rare)
    merged = merged.dropna(subset=["Factuality Score", "Perception Score", "Stereotype Score", "Decisionmaking Score"], how="all")

    # Save merged output
    if save:
        out_path = os.path.join(base_dir, f"{axis}_merged_scores.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"✅ Merged and normalized scores saved to {out_path}")

    return merged


def merge_multiple_axes(base_dir="intermediate"):
    axes = [
        "ability", "age", "nationality", "race_ethnicity_color",
        "physical_traits", "religion", "socioeconomic", "gender_and_sex"
    ]

    for axis in axes:
        print(f"🔄 Processing: {axis}")
        merge_task_scores(axis=axis, base_dir=base_dir, save=True)


def plot_correlation_heatmaps(base_dir="intermediate", out_dir="figuresforpaper/correlation/deepseek", method="spearman"):
    import os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr
    from itertools import combinations

    os.makedirs(out_dir, exist_ok=True)

    axes = [
        "ability", "age", "nationality", "race_ethnicity_color",
        "physical_traits", "religion", "socioeconomic", "gender_and_sex"
    ]

    for axis in axes:
        file_path = os.path.join(base_dir, f"{axis}_merged_scores.csv")
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)

        df = df.rename(columns={
            "Factuality Score": "Factuality",
            "Perception Score": "Perception",
            "Stereotype Score": "Stereotype",
            "Decisionmaking Score": "Decision"
        })
        score_cols = ["Factuality", "Perception", "Stereotype", "Decision"]


        data = df[score_cols].dropna()

        # Correlation matrix
        corr = data.corr(method=method)

        # Significance matrix
        pvals = pd.DataFrame(index=score_cols, columns=score_cols)
        for col1, col2 in combinations(score_cols, 2):
            x, y = data[col1], data[col2]
            try:
                if method == "spearman":
                    _, p = spearmanr(x, y)
                else:
                    _, p = pearsonr(x, y)
            except:
                p = np.nan
            pvals.loc[col1, col2] = p
            pvals.loc[col2, col1] = p
        np.fill_diagonal(pvals.values, np.nan)

        # Plot
        plt.figure(figsize=(8, 8))
        annot = corr.round(2).astype(str)

        # Bold if significant
        for i in range(len(score_cols)):
            for j in range(len(score_cols)):
                if i != j:
                    pval = pvals.iloc[i, j]
                    # if pd.notna(pval) and pval < 0.05:
                    #     annot.iloc[i, j] = f"$\\bf{{{annot.iloc[i, j]}}}$"

        mask = np.triu(np.ones_like(corr, dtype=bool))

                # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))

        # Base: light gray background with colorbar
        base = sns.heatmap(corr, cmap="Greys", cbar=True, square=True,
                        vmin=-1, vmax=1, linewidths=1, linecolor='black',
                        mask=~mask, alpha=0.2,
                        cbar_kws={"pad": 0.02, "shrink": 0.65}, ax=ax)

        # Top: color for lower triangle
        sns.heatmap(corr, mask=mask, annot=annot, fmt="", cmap=paired_cmap, center=0,
                    linewidths=1, linecolor='black',
                    cbar=False,
                    annot_kws={"size": 20}, square=True, vmin=-1, vmax=1, ax=ax)

        # Fix colorbar ticks
        if base.collections and base.collections[0].colorbar:
            base.collections[0].colorbar.ax.tick_params(labelsize=20, width=3)


        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=20, fontweight='medium')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20, fontweight='medium')
        ax.tick_params(axis='x', width=2, pad=5)
        ax.tick_params(axis='y', width=2)
        ax.set_ylabel("")
        ax.collections[0].colorbar.ax.tick_params(labelsize=20, width=3)

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        plt.tight_layout()

        # Save
        out_path = os.path.join(out_dir, f"{axis}_correlation_heatmap.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")

def plot_descriptor_task_heatmap(axis, base_dir="intermediate", out_dir="figuresforpaper/descriptor_matrix/deepseek"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(out_dir, exist_ok=True)

    file_path = os.path.join(base_dir, f"{axis}_merged_scores.csv")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Sort descriptors by factuality
    df = df.sort_values("Factuality Score", ascending=False)

    # Prepare matrix
    score_matrix = df.set_index("Descriptor")[[
        "Factuality Score", "Perception Score", "Stereotype Score", "Decisionmaking Score"
    ]]
    score_matrix.columns = [col.replace(" Score", "") for col in score_matrix.columns]
    score_matrix.columns = [col.replace("making", "") for col in score_matrix.columns]

    # Clean up labels
    score_matrix.index = score_matrix.index.str.replace(" person", "", regex=False)

    # Dynamic figure size based on axis
    # axis_height = {
    #     "ability": 14, "age": 14, "nationality": 30, "race_ethnicity_color": 20,
    #     "religion": 16, "physical_traits": 40, "socioeconomic": 50, "gender_and_sex": 8
    # }
    # height = axis_height.get(axis, max(10, 0.4 * len(score_matrix)))
    # width = max(6, 0.4 * len(score_matrix.columns))

    # plt.figure(figsize=(width, height))

    num_rows = len(score_matrix)
    num_cols = len(score_matrix.columns)
    plt.figure(figsize=(max(8, 0.6 * num_cols), max(8, 0.6 * num_rows)))

    ax = sns.heatmap(score_matrix, cmap=paired_cmap, center=50, annot=True, fmt=".1f",
                     linewidths=1, linecolor='black', cbar_kws={"pad": 0.02},
                     annot_kws={"size": 16})

    # Styling
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20, fontweight='medium')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20, fontweight='medium')
    ax.tick_params(axis='x', width=2, pad=5)
    ax.tick_params(axis='y', width=2)
    ax.set_ylabel("")
    ax.collections[0].colorbar.ax.tick_params(labelsize=20, width=3)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

    # Save
    out_path = os.path.join(out_dir, f"{axis}_descriptor_task_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_path}")



def plot_descriptor_task_heatmaps_all(base_dir="intermediate", out_dir="figuresforpaper/descriptor_matrix/deepseek"):
    import os

    axes = [
        "ability", "age", "nationality", "race_ethnicity_color",
        "physical_traits", "religion", "socioeconomic", "gender_and_sex"
    ]

    for axis in axes:
        print(f"📊 Generating heatmap for: {axis}")
        try:
            plot_descriptor_task_heatmap(axis, base_dir=base_dir, out_dir=out_dir)
        except Exception as e:
            print(f"❌ Failed for {axis}: {e}")


def plot_split_descriptor_heatmaps(axis="nationality", base_dir="intermediate", out_dir="figuresforpaper/descriptor_matrix/deepseek"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    from matplotlib.colors import ListedColormap
    global paired_cmap  # assumes this colormap is already defined
    if "paired_cmap" not in globals():
        colors = ["#C3B1E1", "#008080"]
        paired_cmap = ListedColormap(colors)

    os.makedirs(out_dir, exist_ok=True)

    file_path = os.path.join(base_dir, f"{axis}_merged_scores.csv")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Standardize columns
    df = df.rename(columns={
        "Factuality Score": "Factuality",
        "Perception Score": "Perception",
        "Stereotype Score": "Stereotype",
        "Decisionmaking Score": "Decision"
    })
    df["Descriptor"] = df["Descriptor"].str.replace(" person", "", regex=False).str.strip()

    # Define groups
    consistent_list = [
        "mongolian", "irish", "american", "iraqi", "thai", "scottish",
        "middle eastern", "phillipines", "korean", "african", "chinese"
    ]
    conflicting_list = [
        "russian", "french", "vietnamese", "italian", "mexican",
        "indian", "japanese", "british", "middle eastern",
        "native american", "german", "greek"
    ] #"european", "hispanic", "eskimo", 

    df_consistent = df[df["Descriptor"].isin(consistent_list)].set_index("Descriptor")[["Factuality", "Perception", "Stereotype", "Decision"]]
    df_conflicting = df[df["Descriptor"].isin(conflicting_list)].set_index("Descriptor")[["Factuality", "Perception", "Stereotype", "Decision"]]

    def plot_heatmap(data, label):
        num_rows, num_cols = data.shape
        plt.figure(figsize=(max(8, 0.6 * num_cols), max(8, 0.6 * num_rows)))

        ax = sns.heatmap(data, cmap=paired_cmap, center=50, annot=True, fmt=".1f",
                         linewidths=1, linecolor='black', cbar_kws={"pad": 0.02},
                         annot_kws={"size": 16})

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20, fontweight='medium')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20, fontweight='medium')
        ax.tick_params(axis='x', width=2, pad=5)
        ax.tick_params(axis='y', width=2)
        ax.set_ylabel("")
        ax.collections[0].colorbar.ax.tick_params(labelsize=20, width=3)

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{axis}_{label}_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {out_path}")

    plot_heatmap(df_consistent, "consistent")
    plot_heatmap(df_conflicting, "conflicting")

def plot_cross_model_spiderplots(base_dir="intermediate", output_dir="figuresforpaper/spider_plots"):
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "llava": "#E0115F",     # coral red
        "llama": "#702963",     # soft blue
        "deepseek": "#008080"   # fresh green
    }

    axes = [
        "ability", "age", "gender_and_sex", "nationality",
        "physical_traits", "race_ethnicity_color", "religion", "socioeconomic"
    ]
    axes_labels = [
        "Ability", "Age", "Gender", "Nationality",
        "Physical Traits", "Race", "Religion", "Socioeconomic"
    ]
    tasks = {
        "factuality": "Factuality Score",
        "perception": "Perception Score",
        "stereotype": "Stereotype Score",
        "decisionmaking": "Decisionmaking Score"
    }

    for task, colname in tasks.items():
        raw_scores = pd.DataFrame(index=axes)

        for model in models:
            model_scores = []
            for axis in axes:
                file = os.path.join(base_dir, model, f"{axis}_{task}_scores.csv")
                if os.path.exists(file):
                    df = pd.read_csv(file)
                    df.columns = df.columns.str.strip()
                    value = df[colname].median() if colname in df.columns else np.nan
                else:
                    value = np.nan
                model_scores.append(value)
            raw_scores[model] = model_scores

        norm_scores = raw_scores.copy()
        if task == "perception":
            norm_scores = 100 - norm_scores
        elif task == "stereotype":
            min_val = raw_scores.min().min()
            max_val = raw_scores.max().max()
            if max_val != min_val:
                norm_scores = 100 * (raw_scores - min_val) / (max_val - min_val)
            else:
                norm_scores[:] = 50

        labels = axes_labels
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('white')

        # light teal background
        # ax.set_facecolor("#e0f7f5")

        for model, color in models.items():
            values = norm_scores.loc[:, model].tolist()
            values += values[:1]
            ax.plot(angles, values, label=model, linewidth=3.5, color=color, marker='o', markersize=15)
            ax.fill(angles, values, alpha=0.3, color=color)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=30, fontweight='medium')
        for label in ax.get_xticklabels():
            x, y = label.get_position()
            label.set_position((x, y + 0.18))  # shift radial position inward

        ax.set_ylim(0, 100)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)
        ax.tick_params(axis='y', labelsize=12, width=2)
        # ax.set_title(f"{task.title()}", fontsize=35, weight='bold', pad=25)
        pretty_name = "Decision Making" if task == "decisionmaking" else task.title()
        ax.set_title(pretty_name, fontsize=35, weight='bold', pad=25)
        
        ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.6)
        # ax.spines['polar'].set_linewidth(2)
        ax.spines['polar'].set_color("lightgray")
        ax.spines['polar'].set_linestyle("--")
        ax.spines['polar'].set_linewidth(1)
        # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=14, frameon=False)
        # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=30, frameon=False)


        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{task}_spider_plot.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print("✅ Spider plots with styled background and labeled axes saved.")



def plot_dumbbell_plots(base_dir="intermediate", out_dir="figuresforpaper/dumbbell_plots"):

    os.makedirs(out_dir, exist_ok=True)

    models = ["llava", "llama", "deepseek"]
    # axes = [
    #     "ability", "age", "gender_and_sex", "nationality",
    #     "physical_traits", "race_ethnicity_color", "religion", "socioeconomic"
    # ]
    axes = [
        "nationality"
    ]
    tasks = {
        "factuality": "Factuality Score",
        "perception": "Perception Score",
        "stereotype": "Stereotype Score",
        "decisionmaking": "Decisionmaking Score"
    }

    for axis in axes:
        descriptor_order = None  # ⬅️ declare once per axis

        for task, colname in tasks.items():

            # descriptor_df = pd.DataFrame()
            descriptor_df = None

            for model in models:
                file = os.path.join(base_dir, model, f"{axis}_{task}_scores.csv")
                if not os.path.exists(file):
                    print(f"❌ Missing file: {file}")
                    continue

                df = pd.read_csv(file)
                df.columns = df.columns.str.strip()

                # Standardize column
                if colname not in df.columns:
                    print(f"⚠️ Column not found: {colname} in {file}")
                    continue

                # Choose descriptor column
                descriptor_col = None
                if "Descriptor" in df.columns:
                    descriptor_col = "Descriptor"
                elif "Activity" in df.columns:
                    descriptor_col = "Activity"
                else:
                    continue

                df = df[[descriptor_col, colname]].copy()
                df = df.rename(columns={descriptor_col: "Descriptor", colname: model})
                # descriptor_df = pd.merge(descriptor_df, df, on="Descriptor", how="outer")
                if descriptor_df is None:
                    descriptor_df = df  # first valid df
                else:
                    descriptor_df = pd.merge(descriptor_df, df, on="Descriptor", how="outer")


            if descriptor_df.empty or descriptor_df.shape[1] < 4:
                continue

            # Clean label
            descriptor_df["Descriptor"] = descriptor_df["Descriptor"].str.replace(" person", "", regex=False).str.strip()
            if axis == "nationality":
                descriptor_df["Descriptor"] = descriptor_df["Descriptor"].replace({
                    "native american": "native amer.",
                    "middle eastern": "middle east"
                })


            # Normalize perception (invert)
            if task == "perception":
                for model in models:
                    if model in descriptor_df.columns:
                        descriptor_df[model] = 100 - descriptor_df[model]

            # Normalize stereotype (min-max scaling)
            if task == "stereotype":
                values = descriptor_df[models].values.flatten()
                min_s, max_s = np.nanmin(values), np.nanmax(values)
                for model in models:
                    descriptor_df[model] = 100 * (descriptor_df[model] - min_s) / (max_s - min_s) if max_s != min_s else 50

            # Melt for plotting
            melted = descriptor_df.melt(id_vars="Descriptor", value_vars=models, var_name="Model", value_name="Score")
            if descriptor_order is None:
                descriptor_order = descriptor_df.set_index("Descriptor")[models].mean(axis=1).sort_values().index


            plt.figure(figsize=(10, max(6, 0.4 * len(descriptor_order))))
            ax = plt.gca()
            plt.axvspan(0, 33, color="#702963", alpha=0.1, zorder=0)    # low scores purple #f0e0f7
            plt.axvspan(33, 66, color="#fef4e8", alpha=0.3, zorder=0)   # mid scores peach #fef4e8
            plt.axvspan(66, 100, color="#008080", alpha=0.1, zorder=0)  # high scores green #e0f7f5


            model_colors = {
                "llava": "#E0115F",     # purple
                "llama": "#702963",     # soft blue
                "deepseek": "#008080"   # green
            }

            for i, descriptor in enumerate(descriptor_order):
                row = melted[melted["Descriptor"] == descriptor]
                for model in models:
                    x = row[row["Model"] == model]["Score"].values
                    if len(x) > 0:
                        plt.plot(x[0], i, 'o', color=model_colors[model], markersize=15)
                # print(f"\nDescriptor: {descriptor}\nRow:\n{row}")
                scores = row.set_index("Model").loc[models]["Score"].values
                # plt.plot(scores, [i]*len(scores), '-', color="lightgray", linewidth=5, zorder=0)
                plt.plot(scores, [i]*len(scores), '-', color="lightgray", linewidth=5, zorder=1)
                plt.plot(x[0], i, 'o', color=model_colors[model], markersize=15, zorder=2)




            plt.yticks(range(len(descriptor_order)), descriptor_order, fontsize=22)
            plt.xticks(fontsize=22)
            # plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.xlabel("", fontsize=22)
            # plt.title(f"{axis.replace('_', ' ').title()} – {task.title()}", fontsize=20, weight='bold')
            plt.title(f"{task.title()}", fontsize=30, weight='bold')
            # plt.legend(models, loc="lower right", fontsize=10, frameon=False)
            handles = [mlines.Line2D([], [], color=model_colors[m], marker='o', linestyle='', markersize=10, label=m) for m in models]
            plt.legend(handles=handles, loc="lower right", fontsize=15, frameon=False)

            out_file = os.path.join(out_dir, f"{axis}_{task}_dumbbell.png")
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"✅ Saved: {out_file}")


# factuality_df = compute_factuality_scores(
#     input_dir=f"../../outputs/factuality/deepseek/{axis}",
#     setting=1,
#     save_path=f"intermediate/deepseek/{axis}_factuality_scores.csv"
# )

# perception_df = compute_perception_scores(
#     male_file=f"../../outputs/assumptions/deepseek/{axis}/setting1_2a_{axis}_male_deepseek_responses.csv",
#     female_file=f"../../outputs/assumptions/deepseek/{axis}/setting1_2a_{axis}_female_deepseek_responses.csv",
#     save_path=f"intermediate/deepseek/{axis}_perception_scores.csv"
# )

# stereotype_df = compute_stereotype_scores(
#     male_file=f"../../outputs/socialassumptions/deepseek/{axis}/{axis}_male_1a_deepseek_responses.csv",
#     female_file=f"../../outputs/socialassumptions/deepseek/{axis}/{axis}_female_1a_deepseek_responses.csv",
#     save_path=f"intermediate/deepseek/{axis}_stereotype_scores.csv"
# )

# decision_df = compute_decisionmaking_scores(
#     male_file=f"../../outputs/decisionmaking/deepseek/{axis}/{axis}_male_1a_deepseek_responses.csv",
#     female_file=f"../../outputs/decisionmaking/deepseek/{axis}/{axis}_female_1a_deepseek_responses.csv",
#     save_path=f"intermediate/deepseek/{axis}_decisionmaking_scores.csv"
# )


# merge_multiple_axes()

# plot_correlation_heatmaps()

# plot_descriptor_task_heatmaps_all()

# plot_split_descriptor_heatmaps()

# plot_cross_model_spiderplots()

plot_dumbbell_plots()
