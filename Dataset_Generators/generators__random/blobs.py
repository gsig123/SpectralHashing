from sklearn import manifold, datasets
import argparse
import numpy as np
import matplotlib.pyplot as plt



def plot_2D(dataset_1, color="teal"):
    fig = plt.figure(figsize=(5, 5))
    point_size = 3

    # ---- First subplot
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
    parser.add_argument("-seed", help="seed/random state", type=int, nargs='?', default=None)
    parser.add_argument("-blobs", help="number of blobs", type=int, nargs='?', const=1, default=1)
    parser.add_argument("-blob_std", help="stddev of each blob", type=float, nargs='?', const=1, default=1)
    parser.add_argument("-output", help="blobs_dataset output file", required=1)
    parser.add_argument("-if_color_col", help="color column attached when saving the generated data, where 0 means no and 1 means yes", type=int, required=1)
    args = parser.parse_args()

    return args


def main_blobs():
    args = read_args()

    n_samples = args.n
    n_dimensions = args.d
    seed = args.seed
    blobs = args.blobs
    blobs_std = args.blob_std
    output_file = args.output
    if_color_col = args.if_color_col
    dataset_blobs, color = datasets.samples_generator.make_blobs(
                                                            n_samples=n_samples,
                                                            n_features=n_dimensions,
                                                            cluster_std=blobs_std,
                                                            centers=blobs,
                                                            center_box=(-100.0, 100.0),
                                                            shuffle=True,
                                                            random_state=seed)

    color_col = np.array([[x] for x in color])
    dataset_blobs_with_color_col = np.concatenate((dataset_blobs, color_col), axis=1)

    if if_color_col == 1:
        np.savetxt(output_file, dataset_blobs_with_color_col, delimiter=' ')
    else:
        np.savetxt(output_file, dataset_blobs, delimiter=' ')

    plot_2D(dataset_blobs)

if __name__ == '__main__':
    main_blobs()
