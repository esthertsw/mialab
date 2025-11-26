import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import pandas as pd


def main(result_dir, metric_name):
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

    if metric_name not in df.columns:
        print(f"Error: Metric '{metric_name}' not found in CSV columns: {df.columns.tolist()}")
        return
    
    labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
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
        help='Name of metric to plot'
    )

    args = parser.parse_args()
    main(args.result_dir, args.metric)

