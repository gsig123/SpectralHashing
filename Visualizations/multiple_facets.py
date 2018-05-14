import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_sample_size_vs_f1_for_diff_models():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
    cols = ['Dataset', 'Training size', 'max(F1)', 'Number of bits']
    data_file = 'Data/sample_size_vs_f1_only_model_32.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    print("data df", type(data.iloc[0,1]))
    print(data)



    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(data, col="Dataset", hue="Dataset", col_wrap=2, size=4)


    # Draw a line plot to show the trajectory of each random walk
    grid.map(sns.pointplot, "Training size", "max(F1)", marker="o", ci=None)

    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(6),
             yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             xlim=(-0.5, 4.5),
             ylim=(0.0, 0.6))

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=2)
    plt.show()



def plot_f1_vs_num_bits_for_diff_datasets():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
    sns.set_style("darkgrid", {"axes.linewidth": .000001})
    cols = ['Dataset', 'SH variant', 'Max cuts per PC', 'Number of bits', 'Hamming ball', 'Precision', 'Recall', 'max(F1)', 'Training size', 'k']
    # data_file = 'Data/comparison_all_scores__profi-and-sift.txt'
    data_file = 'Data/comparison_all_scores__mnist-and-profijl.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    print(data)



    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(data, row="Dataset", hue="SH variant", size=5)

    # Draw a line plot to show the trajectory of each random walk
    grid = (grid.map(sns.pointplot, "Number of bits", "max(F1)", hue=data.iloc[:, 1], palette="muted", markers=["*", "*", "*", "*", "*", "*"]).add_legend())

    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(6),
             yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
             xlim=(-0.5, 5.5),
             ylim=(0.0, 0.7))

    grid.fig.tight_layout(rect=[0, 0, 0.8, 0.9], h_pad=2)


    plt.show()


def plot_num_cuts_vs_num_bits_for_diff_datasets():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
    sns.set_style("darkgrid", {"axes.linewidth": .000001})
    cols = ['Dataset', 'SH variant', 'Max cuts per PC', 'Number of bits', 'Hamming ball', 'Precision', 'Recall', 'max(F1)', 'Training size', 'k']
    data_file = 'Data/comparison_all_scores.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    print(data)



    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(data, col="Number of bits", size=5, col_wrap=2)

    # Draw a line plot to show the trajectory of each random walk
    grid = (grid.map(sns.pointplot, "Dataset", "Max cuts per PC", palette="muted").add_legend())


    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(4),
             yticks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
             xlim=(-0.5, 3.5),
             ylim=(0, 8))

    grid.fig.tight_layout(h_pad=2)


    plt.show()


def plot_num_bits_vs_hamm_ball_of_max_F1_for_diff_datasets():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
    sns.set_style("darkgrid", {"axes.linewidth": .000001})
    cols = ['Dataset', 'SH variant', 'Max cuts per PC', 'Number of bits', 'Hamming ball', 'Precision', 'Recall',
            'max(F1)', 'Training size', 'k']
    # data_file = 'Data/comparison_all_scores.txt'
    # data_file = 'Data/comparison_all_scores__profi-and-sift.txt'
    data_file = 'Data/comparison_all_scores__mnist-and-profijl.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    print(data)

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(data, row="Dataset", hue="SH variant", size=5)

    # Draw a line plot to show the trajectory of each random walk
    grid = (grid.map(sns.pointplot, "Number of bits", "Hamming ball", hue=data.iloc[:, 1], palette="muted",
                     markers=["*", "*", "*", "*", "*", "*"]).add_legend())


    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(6),
             yticks=[0, 20, 40, 60, 80, 100, 120],
             xlim=(-0.5, 5.5),
             ylim=(0.0, 120.0))

    grid.fig.tight_layout(rect=[0, 0, 0.8, 0.9], h_pad=2)

    plt.show()


def plot_num_bits_vs_precision_or_recall_at_fixed_hamming_ball():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
    sns.set_style("darkgrid", {"axes.linewidth": .000001})
    cols = ['Dataset', 'SH variant', 'Hamming ball', 'Number of bits', 'Precision', 'Recall', 'max(F1)', 'Training size', 'k']
    data_file = 'Data/comparison_all_scores_num_bits_vs_precision_or_recall_at_fixed_hamming_ball.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    print(data)



    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(data, col="Dataset", size=5, col_wrap=2)

    # Draw a line plot to show the trajectory of each random walk
    grid = (grid.map(sns.barplot, "Number of bits", "Precision", hue=data.iloc[:, 1], palette="muted").add_legend())


    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(6),
             yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             xlim=(-0.5, 5.5),
             ylim=(0, 1))

    grid.fig.tight_layout(rect=[0, 0, 0.8, 0.9], h_pad=2)


    plt.show()



def main():

    # plot_sample_size_vs_f1_for_diff_models()


    # plot_f1_vs_num_bits_for_diff_datasets()
    # plot_num_cuts_vs_num_bits_for_diff_datasets()
    # plot_num_bits_vs_hamm_ball_of_max_F1_for_diff_datasets()
    plot_num_bits_vs_precision_or_recall_at_fixed_hamming_ball()



if __name__ == '__main__':
    main()
