import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import wilcoxon
import argparse

def to_matrix(df, metric="DICE"):
    """Return matrix: rows=labels, columns=[MEAN values] for a metric"""
    d = df[df["METRIC"] == metric]
    pivot = d.pivot(index="LABEL", columns="STATISTIC", values="VALUE")
    return pivot["MEAN"]

def stats_pair(df1, df2, metric):
    x = to_matrix(df1, metric)
    y = to_matrix(df2, metric)
    stat, p = wilcoxon(x, y)
    return stat, p

def stats(metric):

    df_normal   = pd.read_csv("mia-result/Standard 12-02/results_summary.csv", sep=';')
    df_balanced = pd.read_csv("mia-result/Balanced 12-02/results_summary.csv", sep=';')
    df_weighted = pd.read_csv("mia-result/Exagerated Imbalance 12-02/results_summary.csv", sep=';')

    models = {
        "Normal": df_normal,
        "Balanced": df_balanced,
        "Weighted": df_weighted
    }

    if not os.path.isdir("stats"):
        os.makedirs("stats")

    plt.figure(figsize=(10,5))
    for name, df in models.items():
        vals = to_matrix(df, metric)
        plt.plot(vals.index, vals.values, marker='o', label=name)

    plt.xticks(rotation=45)
    plt.ylabel(f"{metric} (Mean)")
    plt.title(f"Per-label {metric} Across Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"stats/per_label_comparison_{metric}.png")
    plt.close()

    overall = {}
    for name, df in models.items():
        overall[name] = to_matrix(df, metric).mean()

    overall_df = pd.DataFrame.from_dict(overall, orient="index", columns=[f"Macro-{metric}"])
    print("\n=== Macro-level Performance ===")
    print(overall_df)

    overall_df.plot(kind="bar", legend=False, title=f"Overall Macro-{metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"stats/overall_macro_comparison{metric}.png")
    plt.close()

    comparisons = [
        ("Normal", "Balanced"),
        ("Normal", "Weighted"),
        ("Balanced", "Weighted")
    ]

    print("\n=== Wilcoxon Signed-Rank Tests Per-Label ===")
    for a, b in comparisons:
        stat, p = stats_pair(models[a], models[b], metric)
        print(f"{a} vs {b}: stat={stat:.3f}, p={p:.4f}")

    #heatmap with absolute values
    diff = (to_matrix(df_balanced, metric) - to_matrix(df_normal, metric)).to_frame("Diff")
    plt.figure(figsize=(6,4))
    sns.heatmap(diff, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Balanced - Normal ({metric})")
    plt.tight_layout()
    plt.savefig(f"stats/heatmap_balanced_vs_normal_{metric}.png")
    plt.close()

    #normalised heatmap
    mvals = pd.DataFrame({name: to_matrix(df, metric) for name, df in models.items()})
    zvals = mvals.apply(lambda row: (row - row.mean()) / row.std(), axis=1) #z-score normalisation
    plt.figure(figsize=(8, 6))
    sns.heatmap(zvals, cmap="vlag", center=0, linewidths=0.5, annot=False)
    plt.title(f"Z-score Heatmap Across Models – {metric}")
    plt.xlabel("Model")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(f"stats/zscore_heatmap_{metric}.png")
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plots results from MIA Pipeline')

    parser.add_argument(
        '--metric',
        type=str,
        help='Name of metric(s) to plot (separate with comma if multiple. e.g. DICE,JACRD)'
    )

    args = parser.parse_args()
    stats(args.metric)
