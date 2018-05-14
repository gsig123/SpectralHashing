import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_f1_hbd_different_datasets():
    # plt.figure(figsize=(16, 9))
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'Hamming ball distance', 'F1-score']

    data_file = 'Data/f1_hammball_diff-datasets.txt'

    data = pd.read_csv(data_file, delimiter=" ")
    data.columns = cols
    print(data)
    print(data.columns)
    g = sns.pointplot(x=data.iloc[:, 1], y=data.iloc[:, 2], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Hamming ball distance', ylabel='F1-score')

    visible_labels = [lab for lab in g.get_xticklabels() if lab.get_visible() is True and lab.get_text() != '']
    plt.setp(visible_labels[::2], visible=False)

    sns.despine(left=True)

    plt.show()

def plot_hammingball_numofbits_optimality_different_datasets():
    # plt.figure(figsize=(16, 9))
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'Average Hamming ball distance',  'Number of bits']

    data_file = 'Data/hammball_numofbits_optimality.txt'

    data = pd.read_csv(data_file, delimiter=" ")
    data.columns = cols
    print(data)
    print(data.columns)
    g = sns.pointplot(x=data.iloc[:, 2], y=data.iloc[:, 1], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel=cols[2], ylabel=cols[1])


    sns.despine(left=True)

    plt.show()


def plot_eval_type_0_precision_recall_hbd():
    # plt.figure(figsize=(16, 9))
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    col_types = [int, int, float, float]

    cols = ['Number of bits', 'Hamming ball', 'precision', 'recall']

    data_file = 'Data/eval-type-0.p-r-hbd.txt'

    data = pd.read_csv(data_file, delimiter=" ")
    data.columns = cols
    print(data)
    print(data.columns)
    g = sns.pointplot(x=data.iloc[:, 1], y=data.iloc[:, 2], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Hamming ball distance', ylabel='Precision')
    g = sns.pointplot(x=data.iloc[:, 1], y=data.iloc[:, 3], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Hamming ball distance', ylabel='Recall')
    # plt.legend(bbox_to_anchor=(1,1), loc=0,
    #            ncol=1, borderaxespad=0.)
    # plt.legend(bbox_to_anchor=(1.015, 0.9), loc=1,bbox_transform=plt.gcf().transFigure)
    # handles, labels = g.get_legend_handles_labels()
    # g.legend(handles, labels)
    # g.legend_.remove()
    sns.despine(left=True)

    plt.show()

def plot_scores_vanilla():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'Number of bits', 'Hamming ball distance', 'Precision', 'Recall', 'max(F1-score)', 'k']

    data_file = 'Data/nbits_f1_vanilla__ss-20000.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 1], y=data.iloc[:, 5], data=data, palette="muted", hue=data.iloc[:, 6])
    g.set(xlabel='Number of bits', ylabel='max(F1-score)')

    sns.despine(left=True)

    plt.show()

def plot_p_r_f1_vanilla():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Score type', 'Number of bits', 'Scores']

    data_file = 'Data/pr_rec_f1__vanilla_ss-20000_k-100__profiset.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 1], y=data.iloc[:, 2], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Number of bits', ylabel='Scores')

    sns.despine(left=True)

    plt.show()


def plot_nbits_f1_vanilla_diff_datasets():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'Number of bits', 'Score', 'Score type label']

    data_file = 'Data/nbits_scores_vanilla__diff-datasets_ss-20000_k-100.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 1], y=data.iloc[:, 2], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Number of bits', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()


def plot_ss_f1_for_diff_num_of_bits():

    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['SH variant', 'Dataset', 'Number of bits', 'Sample size', 'max(F1)', 'k']

    data_file = 'Data/ss_f1_for-diff-num-bits.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 3], y=data.iloc[:, 4], data=data, palette="muted", hue=data.iloc[:, 2])
    g.set(xlabel='Sample size', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()



def plot_num_of_bits_vs_f1__dataset(data_file):

    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'SH variant', 'Number of bits', 'Hamming ball', 'Precision', 'Recall', 'max(F1)', 'k', 'Sample size']


    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 2], y=data.iloc[:, 6], data=data, palette="muted", hue=data.iloc[:, 1])
    g.set(xlabel='Number of bits', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()


def plot_dim__d_vs_f1__for_diff_models():

    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'SH variant', 'Dimensionality', 'Number of bits', 'max(F1)', 'k', 'Sample size']

    data_file = 'Data/clustered_diff-dim__d_vs_f1__for_diff_models.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 2], y=data.iloc[:, 4], data=data, palette="muted", hue=data.iloc[:, 3])
    g.set(xlabel='Dimensionality', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()


def plot_clustered_numclusters_vs_f1__for_diff_models():

    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'SH variant', 'Dimensionality', 'Number of bits', 'Number of clusters', 'max(F1)', 'k', 'Training size']

    data_file = 'Data/clustered_num-clusters_vs_f1__for_diff_models.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 4], y=data.iloc[:, 5], data=data, palette="muted", hue=data.iloc[:, 3])
    g.set(xlabel='Number of clusters', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()


def plot_vanilla_related_num_bits_f1():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'SH variant', 'Dimensionality', 'Number of bits', 'Number of clusters', 'max(F1)', 'k',
            'Testing size']

    data_file = 'Data/clustered_num-clusters_vs_f1__for_diff_models.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 4], y=data.iloc[:, 5], data=data, palette="muted", hue=data.iloc[:, 3])
    g.set(xlabel='Number of clusters', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()


def plot_paper_results_reconstruction_MNIST():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'SH variant', 'Number of bits', 'Prop. of good neighbors with Hamm. distance <= 3', 'k', 'Testing size']

    data_file = 'Data/paper_results_reconstruction_MNIST.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 2], y=data.iloc[:, 3], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Number of bits', ylabel='Prop. of good neighbors with Hamm. distance <= 3')

    sns.despine(left=True)

    plt.show()


def plot_vanilla_performance_for_diff_datasets():
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['Dataset', 'SH variant', 'Max cuts per PC', 'Number of bits', 'Hamming ball', 'Precision', 'Recall', 'max(F1)', 'Training size', 'k']

    data_file = 'Data/nbits_scores_vanilla__diff-datasets_ss-30000_k-100.txt'

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 3], y=data.iloc[:, 7], data=data, palette="muted", hue=data.iloc[:, 0])
    g.set(xlabel='Number of bits', ylabel='max(F1)')

    leg = plt.legend(loc='best', borderpad=0.3,
                     shadow=False, markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    sns.despine(left=True)

    plt.show()

def plot_sample_size_vs_f1_for_diff_models(data_file):

    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    cols = ['SH variant', 'Dataset', 'Number of bits', 'Sample size', 'max(F1)', 'k']

    data = pd.read_csv(data_file, delimiter=" ", header=None)
    data.columns = cols
    g = sns.pointplot(x=data.iloc[:, 3], y=data.iloc[:, 4], data=data, palette="muted", hue=data.iloc[:, 2])
    g.set(xlabel='Sample size', ylabel='max(F1)')

    sns.despine(left=True)

    plt.show()

def main():
    # plot_eval_type_0_precision_recall_hbd()
    # plot_f1_hbd_different_datasets()
    # plot_hammingball_numofbits_optimality_different_datasets()
    # plot_scores_vanilla()
    # plot_p_r_f1_vanilla()
    # plot_nbits_f1_vanilla_diff_datasets()

    plot_ss_f1_for_diff_num_of_bits()



    # data_files = [
    #     'Data/num_of_bits_vs_f1__dataset_Profi-set.txt',
    #     'Data/num_of_bits_vs_f1__dataset_SIFT.txt',
    #     'Data/num_of_bits_vs_f1__dataset_MNIST.txt',
    #     'Data/num_of_bits_vs_f1__dataset_Profi-set-JL.txt'
    # ]
    # for data_file in data_files:
    #     plot_num_of_bits_vs_f1__dataset(data_file)


    # plot_dim__d_vs_f1__for_diff_models()


    # plot_clustered_numclusters_vs_f1__for_diff_models()
    
    # plot_vanilla_related_num_bits_f1()

    # plot_paper_results_reconstruction_MNIST()

    # plot_vanilla_performance_for_diff_datasets()


    data_file_whole = 'sample_size_vs_f1_for_diff_models.txt'
    # data_files = [
    #     'Data/ss_f1_for-diff-num-bits__Profi-set.txt',
    #     'Data/ss_f1_for-diff-num-bits__SIFT.txt',
    #     'Data/ss_f1_for-diff-num-bits__MNIST.txt',
    #     'Data/ss_f1_for-diff-num-bits__Profi-set-JL.txt'
    # ]
    # for data_file in data_files:
    #     print(data_file)
        # plot_sample_size_vs_f1_for_diff_models(data_file)

if __name__ == '__main__':
    main()
