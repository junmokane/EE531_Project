import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from collections import defaultdict


def plot_one(exp_names, csv_slices, feature, env_name, re_feature):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.canvas.set_window_title(feature)
    for i, csv_slice in enumerate(csv_slices):
        plt.plot(csv_slice[re_feature[i]].to_numpy(), linewidth=3)
        ax.yaxis.get_offset_text().set_fontsize(30)
    plt.legend(exp_names, fontsize=30, fancybox=True)
    plt.title(env_name, fontsize=30)
    plt.xlabel("iteration", fontsize=30)
    plt.xticks(fontsize=30)
    # plt.ylabel(feature, fontsize=15)
    plt.ylabel('average return', fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()


def plot_data(args):
    path = args.file
    features = args.f
    style = args.s
    
    plt.style.use(style)
    features = features[0].split(",")

    for feature in features:
        path = path.rstrip('/').rstrip('\\')
        env_name = path.split('/')[-1]
        csv_paths = glob.glob(f"{path}/**/progress.csv")
        exp_names = [csv_path.split("/")[-2] for csv_path in csv_paths]
        exp_names_index = np.argsort(exp_names)
        csv_paths = [csv_paths[j] for j in exp_names_index]
        exp_names = [exp_names[j] for j in exp_names_index]

        assert len(csv_paths) > 0, "There is no csv files"

        csv_slices = []
        re_feature = []
        for csv_path in csv_paths:
            csv = pd.read_csv(csv_path)
            key_val = csv.keys().tolist()
            print(csv)
            print(csv.keys().tolist())
            if feature == 'AverageTrainReturn_all_train_tasks' and feature not in key_val:
                if 'train-AverageReturn' in key_val:
                    tmp = 'train-AverageReturn'
                elif 'Step_1-AverageReturn' in key_val:
                    tmp = 'Step_1-AverageReturn'
                else:
                    exit()
            else:
                tmp = feature
            re_feature.append(tmp)
            csv_slices.append(csv.loc[:, [tmp]])
            del csv

        plot_one(exp_names, csv_slices, feature, env_name, re_feature)
    plt.show()


if __name__ == "__main__":
    # To run, refer README.md
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the task directory')
    parser.add_argument('--f', type=str, nargs='+',
                        help='List of features to plot')
    parser.add_argument('--s', type=str, default='ggplot',
                        help='Style of plots, Look at (https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html)')
    args = parser.parse_args()
    plot_data(args)