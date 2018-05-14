"""
Measure the recall and precision of compressors.
Scores are measured by finding the actual k neighbors and checking
the actually returned values
"""

import argparse
import importlib
import math
import numpy as np
from helpers import normalize_data, train_sh

def read_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=1)
    parser.add_argument('-indexed', default=False, type=bool)
    parser.add_argument('-compressor', required=1)
    parser.add_argument('-bits', required=1, type=int)
    parser.add_argument('-samples', default=100, type=int)
    args = parser.parse_args()
    return args

def split_dataset(dataset, num_test):
    """Split dataset into training and test set"""
    indices = np.random.permutation(dataset.shape[0])
    return dataset[indices[num_test:]], dataset[indices[:num_test]]

def compare(actual_indices, expected_indices):
    """Compute recall and precision for the given search result"""
    intersection = np.intersect1d(actual_indices, expected_indices)
    recall = len(intersection)/len(expected_indices) if len(expected_indices) != 0 else math.nan
    precision = len(intersection)/len(actual_indices) if len(actual_indices) != 0 else math.nan
    return recall, precision

def print_results(measured_ks, recall, precision, num_returned, data_dimensionality, 
                                        data_no_points, bucket_sizes):
    """Print the results"""

    col_width = 30
    split_line_width = 120 
    split_string = "-" * split_line_width
    
    print(split_string)
    print(split_string)
    print("Number of points in dataset: {}".format(data_no_points))
    print("Dimensionality of original dataset: {}".format(data_dimensionality))
    print("Bucket Sizes: {}".format(bucket_sizes))
    print(split_string)
    print(split_string)

    headers = ["Hamming Distance", "Recall", "Precision", "# of Points Returned"]
    headers_string = "".join(str(word).ljust(col_width) for word in headers)

    for k_idx, k in enumerate(measured_ks):
        print('K = {}'.format(k))
        print(split_string)
        print(headers_string)
        print(split_string)
        result_matrix = create_result_matrix(k_idx, recall, precision, num_returned)
        for row in result_matrix:
            print("".join(str(value).ljust(col_width) for value in row))
    print(split_string)
    print(split_string)

def create_result_matrix(k_idx, recall, precision, num_returned):
    result_matrix = []
    for dist in range(0, len(recall[k_idx])):
        result_matrix.append(
            [
                dist, 
                np.nanmean(recall[k_idx][dist]), 
                np.nanmean(precision[k_idx][dist]), 
                np.nanmean(num_returned[k_idx][dist])
            ])
    return result_matrix


def check_bucket_sizes(compressed_data):
    """Compute and print the number of buckets of each size"""
    print('Checking bucket sizes')
    bucket_sizes = {}
    for hashcode in compressed_data:
        # Because keys can only be strings
        key = hashcode.tostring()
        if not key in bucket_sizes:
            bucket_sizes[key] = 0
        bucket_sizes[key] += 1
    size_count = {}
    for size in bucket_sizes.values():
        if not size in size_count:
            size_count[size] = 0
        size_count[size] += 1
    bucket_sizes = sorted(size_count.items())
    print('Bucket sizes: {}'.format(bucket_sizes))
    return sorted(bucket_sizes)

def main():
    """Run the test"""
    args = read_args()
    compressor = importlib.import_module(args.compressor)
    print('Reading data')
    # Some of the datasets have row indexes in the first column
    indexed = args.indexed
    bits = args.bits 
    
    if indexed: 
        data = np.genfromtxt(args.data, delimiter=' ', dtype=np.float)[:,1:]
    else: 
        data = np.genfromtxt(args.data, delimiter=' ', dtype=np.float)
    # Basic data about the dataset
    data_dimensionality = len(data[0])
    datasets_no_points = len(data)
    # Normalize the data
    data = normalize_data(data)
    print('Splitting data')
    training_data, test_data = split_dataset(data, args.samples)
    print('Creating model')

    model = train_sh(training_data, bits, args.data, '__test.log')
    print('Compressing dataset')
    _, compressed_data = compressor.compress(
        np.concatenate((test_data, training_data), axis=0),
        model,
        model.training_filename)
    compressed_training_data = compressed_data[args.samples:]
    compressed_test_data = compressed_data[:args.samples]
    # Get the bucket sizes
    bucket_sizes = check_bucket_sizes(compressed_data)
    # values of k used for measuring
    measured_ks = [10, 50, 100, 500]
    print('Computing recall and precision')
    recalls = [[[] for _ in range(0, bits+1)] for _ in measured_ks] # k*dist array of lists
    precisions = [[[] for _ in range(0, bits+1)] for _ in measured_ks] # k*dist array of lists
    num_returned = [[[] for _ in range(0, bits+1)] for _ in measured_ks]
    for idx, hashcode in enumerate(compressed_test_data):
        print("Computing {}".format(idx))
        hamming_dists = [
            sum([bin(xi^hashcode[i]).count('1') for i, xi in enumerate(x)])
            for x in compressed_training_data
        ]
        hamming_indices = np.argsort(hamming_dists)
        real_dists = [np.linalg.norm(x-test_data[idx]) for x in training_data]
        real_indices = np.argsort(real_dists)

        num_found = 0
        for distance in range(0, bits+1):
            while num_found < len(hamming_indices)\
            and hamming_dists[hamming_indices[num_found]] <= distance:
                num_found += 1
            for k_idx, k in enumerate(measured_ks):
                recall, precision = compare(hamming_indices[:num_found], real_indices[:k])
                recalls[k_idx][distance].append(recall)
                precisions[k_idx][distance].append(precision)
                num_returned[k_idx][distance].append(num_found)
    print_results(
        measured_ks, recalls, precisions, 
        num_returned, data_dimensionality, 
        datasets_no_points, bucket_sizes)

bits_list = [16, 32, 64, 128]

main()
