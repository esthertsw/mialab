import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import pandas as pd
import matplotlib.cm as cm
from matplotlib.patches import Patch


def main(result_dir, metric_list, combine_plots=False):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    #pass  # pass is just a placeholder if there is no other code

    script_dir = os.path.dirname(sys.argv[0])
    path = os.path.join(os.getcwd(), 'mia-result', result_dir, 'results.csv')

    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return

    df = pd.read_csv(path, delimiter=';')

    for metric_name in metric_list:
        if metric_name not in df.columns:
            print(f"Error: Metric '{metric_name}' not found in CSV columns: {df.columns.tolist()}")
            return
    
    labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
    
    if combine_plots:
        plt.figure(figsize=(10,6))
        data = []
        positions = []
        current_x = 1

        for label in labels:
            for m, metric_name in enumerate(metric_list):
                arr = df[df["LABEL"] == label][metric_name].values
                data.append(arr)
                positions.append(current_x + m * 0.3)  # small spacing within the pair
            current_x += 1  # big jump to next label

        bp = plt.boxplot(
            data,
            positions=positions,
            widths=0.25,
            patch_artist=True
        )

        # color each box
        colors = cm.tab10(np.linspace(0, 1, len(metric_list)))

        for i, box in enumerate(bp["boxes"]):
            metric_index = i % len(metric_list)
            box.set_facecolor(colors[metric_index])
            box.set_alpha(0.7)
        # label only the *center* of each pair
        tick_positions = [np.mean(positions[i:i+len(metric_list)]) 
                        for i in range(0, len(positions), len(metric_list))]

        plt.xticks(tick_positions, labels)

        plt.grid(axis="y")
        plt.ylabel("Metric value")
        plt.title("Metrics per Brain Structure")
        legend_handles = [Patch(facecolor=colors[i], label=metric_list[i], alpha=0.7) 
                for i in range(len(metric_list))]
        plt.legend(handles=legend_handles, title="Metrics")

        plt.tight_layout()
        plt.show()
        
    else:
        for metric_name in metric_list:
            
            data = [df[df['LABEL'] == label][metric_name].values for label in labels]
            # Plot a boxplot
            plt.figure(figsize=(10,6))
            plt.boxplot(data, tick_labels=labels)
            plt.title(f"{metric_name} per Brain Structure")
            plt.ylabel(metric_name)
            plt.grid(axis='y')
            plt.tight_layout()

            save_path = os.path.join(os.getcwd(), 'mia-result', result_dir, f'{metric_name}_plot.png')
            plt.savefig(save_path)
            plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plots results from MIA Pipeline')

    parser.add_argument(
        '--result_dir',
        type=str,
        help='Name of directory for result in mia-result'
    )

    parser.add_argument(
        '--metric',
        type=str,
        help='Name of metric(s) to plot (separate with comma if multiple. e.g. DICE,JACRD)'
    )

    parser.add_argument(
        '--combine_plots',
        type=str,
        help='Choice of plotting multiple metrics at once [T/F]'
    )

    args = parser.parse_args()
    metrics = ((args.metric).upper()).split(',')
    combine_plots = False
    if len(metrics) > 1 and args.combine_plots is not None:
        combine_plots = True if args.combine_plots.upper() == 'T' else False
    main(args.result_dir, metrics, combine_plots)
