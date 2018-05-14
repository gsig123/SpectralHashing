import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def sample_spherical(n_samples, n_dim):
    vec = np.random.randn(n_dim, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

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

def main_random():
    args = read_args()

    n_samples = args.n
    n_dimensions = args.d
    output_file = args.output

    # -- Make random data around the unit sphere -- #
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    data = sample_spherical(n_samples, n_dimensions).T

    if data.shape[1] > 2:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
        ax.plot_wireframe(x, y, z, color='#939393', rstride=1, cstride=1)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=40, c='teal', zorder=100)

    print(data.shape)
    plot_2D(data)
    plt.show()

    # -- Save data to file -- #
    np.savetxt(output_file, data, delimiter=' ')


main_random()
