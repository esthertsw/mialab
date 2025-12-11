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

    df_normal   = pd.read_csv("mia-result/Standard 12-02/results.csv", sep=';')
    df_balanced = pd.read_csv("mia-result/Balanced 12-02/results.csv", sep=';')
    df_weighted_large = pd.read_csv("mia-result/Weighted_large 12-03/results.csv", sep=';')
    df_weighted_small = pd.read_csv("mia-result/Weighted_small 12-03/results.csv", sep=';')

    
    models = {
        "Normal": df_normal,
        "Balanced": df_balanced,
        "Weighted large": df_weighted_large, 
        "Weighted small" : df_weighted_small,
    }

    if not os.path.isdir("stats"):
        os.makedirs("stats")

    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = dict(zip(models.keys(), palette))

    plt.figure(figsize=(15, 7))

    all_rows = []
    overall_means = {}

    for model_name, df in models.items():
        if metric not in df.columns:
            continue

        overall_means[model_name] = df[metric].mean()

        for label in sorted(df["LABEL"].unique()):
            vals = df[df["LABEL"] == label][metric].values
            for v in vals:
                all_rows.append({
                    "LABEL": label,
                    "VALUE": v,
                    "MODEL": model_name
                })

    plot_df = pd.DataFrame(all_rows)

    sns.boxplot(
        data=plot_df,
        x="LABEL",
        y="VALUE",
        hue="MODEL",
        palette=model_colors,
        width=0.75,
        fliersize=2
    )

    ax = plt.gca()
    xlim = ax.get_xlim()
    for model_name, mean_val in overall_means.items():
        ax.hlines(
            y=mean_val,
            xmin=xlim[0],
            xmax=xlim[1],
            colors=model_colors[model_name],
            linestyles='dashed',
            linewidth=1.5,
            label=f"{model_name} overall"
        )

    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Model")

    plt.xticks(rotation=45)
    plt.title(f"{metric} per Brain Structure (with Overall Mean)")
    plt.ylabel(metric)
    plt.tight_layout()

    save_path = os.path.join("stats", f"{metric}_boxplot.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plots results from MIA Pipeline')

    parser.add_argument(
        '--metric',
        type=str,
        help='Name of metric(s) to plot (separate with comma if multiple. e.g. DICE,JACRD)'
    )

    args = parser.parse_args()
    stats(args.metric)
