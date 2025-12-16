import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import pandas as pd
import matplotlib.cm as cm
from matplotlib.patches import Patch

def main(save_dir, directories, metric_list, combine):
    dataframes = []

    for dir in directories:
        path = os.path.join(os.getcwd(), 'mia-result', dir, 'results.csv')
        if not os.path.exists(path):
            print(f"Error: {path} does not exist.")
            return
        df = pd.read_csv(path, delimiter=';')
        dataframes.append(df)
        
    labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']

    if combine == "metrics":
        # Use the first dataframe (single directory) and plot multiple metrics
        df = dataframes[0]
        plt.figure(figsize=(10,6))
        data = []
        positions = []
        current_x = 1

        for label in labels:
            for m, metric_name in enumerate(metric_list):
                arr = df[df["LABEL"] == label][metric_name].values
                data.append(arr)
                positions.append(current_x + m * 0.3)
            current_x += 1

        bp = plt.boxplot(data, positions=positions, widths=0.25, patch_artist=True)
        colors = cm.tab10(np.linspace(0, 1, len(metric_list)))
        for i, box in enumerate(bp["boxes"]):
            metric_index = i % len(metric_list)
            box.set_facecolor(colors[metric_index])
            box.set_alpha(0.7)

        tick_positions = [np.mean(positions[i:i+len(metric_list)]) 
                        for i in range(0, len(positions), len(metric_list))]
        plt.xticks(tick_positions, labels)
        plt.grid(axis="y")
        plt.ylabel("Metric value")
        plt.title("Metrics per Brain Structure")
        legend_handles = [Patch(facecolor=colors[i], label=metric_list[i], alpha=0.7) 
                for i in range(len(metric_list))]
        plt.legend(handles=legend_handles, title="Metrics")

        os.makedirs(os.path.join(os.getcwd(), 'mia-result', save_dir), exist_ok=True)
        save_path = os.path.join(os.getcwd(), 'mia-result', save_dir, f'{"_".join(metric_list)}_plot.png')
        plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

    elif combine == "directories":
        # Plot the same metric across multiple directories
        metric_name = metric_list[0]
        plt.figure(figsize=(10,6))
        data = []
        positions = []
        current_x = 1

        for label in labels:
            for d, df in enumerate(dataframes):
                arr = df[df["LABEL"] == label][metric_name].values
                data.append(arr)
                positions.append(current_x + d * 0.3)
            current_x += 1

        bp = plt.boxplot(data, positions=positions, widths=0.25, patch_artist=True)
        colors = ["#4C72B0", "#DD8452"]
        for i, box in enumerate(bp["boxes"]):
            dir_index = i % len(dataframes)
            box.set_facecolor(colors[dir_index])
            
        tick_positions = [np.mean(positions[i:i+len(dataframes)]) 
                        for i in range(0, len(positions), len(dataframes))]
        plt.xticks(tick_positions, labels)
        plt.grid(axis="y")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} per Brain Structure")

        legend_handles = [Patch(facecolor=colors[i], label=directories[i], alpha=0.7)
                        for i in range(len(dataframes))]
        plt.legend(handles=legend_handles, title="Directory")

        os.makedirs(os.path.join(os.getcwd(), 'mia-result', save_dir), exist_ok=True)
        save_path = os.path.join(os.getcwd(), 'mia-result', save_dir, f'{metric_name}_directories_plot.png')
        plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

    else:
        # Default: plot single metric in single directory
        df = dataframes[0]
        for metric_name in metric_list:
            data = [df[df['LABEL'] == label][metric_name].values for label in labels]
            plt.figure(figsize=(10,6))
            plt.boxplot(data, labels=labels)
            plt.title(f"{metric_name} per Brain Structure")
            plt.ylabel(metric_name)
            plt.grid(axis='y')
            plt.tight_layout()
            os.makedirs(os.path.join(os.getcwd(), 'mia-result', save_dir), exist_ok=True)
            save_path = os.path.join(os.getcwd(), 'mia-result', save_dir, f'{metric_name}_plot.png')
            plt.savefig(save_path)
            plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plots results from MIA Pipeline')

    parser.add_argument(
        '--result_dir',
        type=str,
        help='Name of directories for result in mia-result (separate with comma if multiple)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        help='Name of metric(s) to plot (separate with comma if multiple. e.g. DICE,JACRD)'
    )

    parser.add_argument(
        '--combine',
        type=str,
        default="false",
        help='Combine metrics or directories (type: metrics or directories)'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='Name of directory to save plot'
    )

    args = parser.parse_args()
    metrics = args.metric.upper().split(',')
    directories = args.result_dir.split(',')

    if args.save_dir is None:
        args.save_dir = directories[0]

    if len(metrics) > 1 and len(directories) > 1:
        print("Unable to plot multiple directories and multiple metrics simultaneously, choose one.")
    else:
        main(args.save_dir, directories, metrics, args.combine)