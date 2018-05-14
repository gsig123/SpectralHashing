"""
Compress the data using a heuristic where the distance between points that are mapped
to the same locations in the principal components is considered.
"""

import math
import numpy as np
import random

NUM_BUCKETS = 2000
# A number where we are certain that there are no values greater than it in the final data.
MAX_VAL = 20.0

def mean_distances(points):
    """
    Calculate the mean distances between the given points.
    May use random sampling to do it.
    """
    mean = 0.0
    for point_a in points:
        point_b = random.choice(points)
        diff = point_a-point_b
        magnitude = np.sqrt(diff.dot(diff))
        mean += magnitude
    input_size = len(points)
    if input_size == 0:
        return 0.0
    return mean/(input_size*input_size)

def bucket(splits, value):
    """Compute the bucket id of the value when buckets are split at the given points. """
    for bucketid, split in enumerate(splits):
        if value < split:
            return bucketid
    return len(splits)

def normalize_sum(values):
    """Normalize the values, such that they sum to 1."""
    total = sum(values)
    return list(map(lambda x: x/total, values))

def create_splits(values, num_buckets):
    """Find split points so that the sum of weights in each bucket is equal."""
    next_split = 1
    cumulative_sum = 0.0
    splits = []
    for idx, value in enumerate(values):
        cumulative_sum += value
        if cumulative_sum > next_split/num_buckets+1e-8: # avoid split at 100%
            splits.append((idx+0.5)/NUM_BUCKETS*MAX_VAL)
            next_split += 1
    return splits

def weigh_bucket(normalized_mean_dist, bucket_size):
    """Weigh a bucket, so that buckets with high distances are weighted low"""
    return bucket_size*(1.0-normalized_mean_dist)

def graycodes(num_bits):
    """A list of all graycodes with the number of bits"""
    if num_bits == 1:
        return [0, 1]
    recursive_codes = graycodes(num_bits-1)
    reversed_codes = list(recursive_codes)
    reversed_codes.reverse()
    return recursive_codes + [x+(1<<(num_bits-1)) for x in reversed_codes]

def assign_hashcodes(num_bits, num_buckets):
    """Assign hash codes to buckets"""
    if num_bits == 0:
        return []
    return graycodes(num_bits)[:num_buckets]

def partition_dimension(data, dimension, num_bits):
    """Partition a data into buckets and assign a hash code to each of them"""
    # Map the points to discrete segments
    buckets = [[] for _ in range(0, NUM_BUCKETS)]
    for point in data:
        bucketid = math.floor(point[dimension]/MAX_VAL*NUM_BUCKETS)
        buckets[bucketid].append(point)

    mean_dists = list(map(mean_distances, buckets))
    max_dist = max(mean_dists)
    normalized_mean_dists = list(map(lambda x: x/max_dist, mean_dists))
    bucket_weights = list(map(lambda bucket: weigh_bucket(*bucket),
                              zip(normalized_mean_dists, map(len, buckets))))
    normalized_weights = normalize_sum(bucket_weights)
    # An estimate on the number of buckets we should probably use.
    num_buckets = 1 << num_bits
    #num_buckets = math.ceil((1 << num_bits)*3.0/4.0)
    splits = create_splits(normalized_weights, num_buckets)
    hashcodes = assign_hashcodes(num_bits, num_buckets)
    return {
        'splits': splits,
        'hashcodes': hashcodes,
        'num_bits': num_bits,
    }

def convert_modes(modes):
    """Convert the modes given by the sh_model to a list of bits for each dimension"""
    return np.count_nonzero(modes, axis=0)

def create_compression_model(data, sh_model):
    """
    Create the model for the compression step.
    This consists of the points where the dimension is split into buckets and the
    hashcodes assigned to each bucket. These values are given for each dimension.
    """
    _, dimensions = data.shape
    bits = convert_modes(sh_model.modes)
    model = [partition_dimension(data, d, bits[d]) for d in range(0, dimensions)]
    return model

def compress(normalized_data, sh_model, _):
    """Compress the data"""
    input_size = len(normalized_data)
    data_pca = normalized_data.dot(sh_model.pc_from_training)
    data = data_pca - np.tile(sh_model.mn, (input_size, 1))

    compression_model = create_compression_model(data, sh_model)

    outputs = []
    for point in data:
        hashcode = []
        for dimension, buckets in enumerate(compression_model):
            if buckets['hashcodes']:
                bucketid = bucket(buckets['splits'], point[dimension])
                print(bucketid, buckets['hashcodes'])
                hashcode.append(buckets['hashcodes'][bucketid])
        outputs.append(hashcode)
    return None, np.array(outputs)
