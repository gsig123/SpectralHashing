"""
Plot the input data with colors according to their assigned hashcodes,
as well as the principal components.
Only works for 2d data
"""
import argparse
import importlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from helpers import normalize_data

def to_binary(val, length):
    """convert a number to a binary string"""
    return bin(val)[2:].rjust(length, '0')

def read_args():
    """Read arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="full path to the visualized model", required=1)
    parser.add_argument("-input", help="full path to the input data", required=1)
    parser.add_argument(
        "-compressor",
        help="A module containing a compress(data_norm, sh_model, label) function",
        default="Compressors.vanilla")
    return parser.parse_args()

def main():
    """Create the plot"""
    args = read_args()
    compressor_file = args.compressor
    compressor = importlib.import_module(compressor_file)
    model = pickle.load(open(args.model, 'rb'))
    input_data = np.genfromtxt(args.input, delimiter=' ', dtype=np.float)

    principal_components = model.pc_from_training
    input_data_norm = normalize_data(input_data)
    _, hash_codes = compressor.compress(input_data_norm, model, model.training_filename)

    # Split depending on assigned hash values
    hash_buckets = [[] for _ in range(0, 2**model.n_bits)]
    for point, hash_val in zip(input_data_norm, hash_codes[:, 0]):
        hash_buckets[hash_val].append(point)

    # Plot differently colored points, depending on hash
    legend_handles = []
    for hash_val, points in enumerate(hash_buckets):
        x_coords = list(map(lambda x: x[0], points))
        y_coords = list(map(lambda x: x[1], points))
        points, = plt.plot(x_coords, y_coords, '.', label=to_binary(hash_val, model.n_bits))
        legend_handles.append(points)

    # Plot principal components
    for principal_component in principal_components:
        # Eigenvectors were flipped during training
        plt.plot([0.5, 0.5+principal_component[1]/2], [0.5, 0.5+principal_component[0]/2], 'r')

    # Show legend outside to the right
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", handles=legend_handles)
    # Ensure there is enough space for the legend
    plt.subplots_adjust(right=0.8)
    plt.show()


main()
