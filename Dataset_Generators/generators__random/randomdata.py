import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_2D(dataset_1, color="teal"):
    fig = plt.figure(figsize=(5, 5))
    point_size = 3

    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        c=color,
        s=point_size)

    plt.show()


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help="n_samples", type=int, required=1)
    parser.add_argument("-d", help="n_dimensions", type=int, required=1)
    parser.add_argument("-output", help="blobs_dataset output file", required=1)
    args = parser.parse_args()

    return args


def main_random():
    args = read_args()

    n_samples = args.n
    n_dimensions = args.d
    output_file = args.output

    # -- Make random data -- #
    random_data = np.random.rand(n_samples, n_dimensions)
    plot_2D(random_data)
    print(random_data.shape)
    np.savetxt(output_file, random_data, delimiter=' ')


main_random()
