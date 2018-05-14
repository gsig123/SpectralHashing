"""
Variation of the median split
"""

from helpers import *
import numpy as np


def convert_modes(modes):
    """Convert the modes given by the sh_model to a list of bits for each dimension"""
    return np.count_nonzero(modes, axis=0)


def find_val_index(data, value):
    """Find the first index that is greater than the supplied value"""
    for i, x in enumerate(data):
        if x > value:
            return i
    return 0


def partition_dimension_rec(data, bits):
    """Partition the data for each of the dimensions"""

    hashes = [[] for _ in range(0, len(data))]

    if bits == 0:
        return hashes

    mean_dimension = data.mean()
    mean_index = find_val_index(data, mean_dimension)

    index_left = int(mean_index / 2)
    index_right = int((mean_index + len(data)) / 2)

    split_left = data[0:index_left]
    split_middle = data[index_left:index_right]
    split_right = data[index_right:]

    hashes_rec = partition_dimension_rec(split_left, bits - 1) + partition_dimension_rec(split_middle,
                                                                                         bits - 1) + partition_dimension_rec(
        split_right, bits - 1)

    index = 0

    for _ in split_left:
        hashes[index] = [0]
        index += 1

    for _ in split_middle:
        hashes[index] = [1]
        index += 1

    for _ in split_right:
        hashes[index] = [0]
        index += 1

    if bits > 1:
        hashes = list(map(lambda a: a[0] + a[1], [list(x) for x in (zip(hashes, hashes_rec))]))

    return hashes


def sort_indexes(input):
    """Sort the array indexes such that each index holds an index of the sorted position of the element"""
    indexes = list(range(len(input)))
    indexes.sort(key=lambda x: input[x])
    output = [0] * len(indexes)

    for i, x in enumerate(indexes):
        output[x] = i

    return output


def compress(normalized_data, sh_model, _):
    """Compress the data"""
    input_size = len(normalized_data)
    data_pca = normalized_data.dot(sh_model.pc_from_training)
    data = data_pca - np.tile(sh_model.mn, (input_size, 1))

    _, dimensions = data.shape
    bits = convert_modes(sh_model.modes)

    compressed = [partition_dimension_rec(np.array(sorted(data[0:, d:d + 1])), bits[d]) for d in range(0, dimensions)]

    merged = [[] for _ in range(0, len(data))]

    for d, compressed_dim in enumerate(compressed):
        sort_idx = sort_indexes(data[0:, d:d + 1])
        for i in range(0, len(compressed_dim)):
            merged[i] = merged[i] + compressed_dim[sort_idx[i]]

    return None, compact_bit_matrix(np.array(merged))
