import random

import numpy as np
import argparse

# -- Usage Example on MNIST case n=1000 -- #
# python sample-data.py -i "./Data/MNIST/mnist.data" -dataset_size 70000 -ss_or_fr "ss" -ss 1000 -ext -num_s=1 ".train"
# python sample-data.py -i "./Data/MNIST/mnist.data" -dataset_size 70000 -ss_or_fr "ss" -ss 1000 -ext ".test"

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file to sample from", required=1)
    parser.add_argument("-dataset_size", help="input dataset size", required=1)
    parser.add_argument("-fr", help="data fraction to sample", type=int, default=10)
    parser.add_argument("-ss", help="sample size (in case we don't sample by fraction of the dataset)", type=int, default=1000)
    parser.add_argument("-seed", help="sampling seed (sampling with probability)", type=int, default=1)
    parser.add_argument("-num_s", help="number of samples, how many samples of teh specific type to generate (generally 5)", type=int, default=5)
    parser.add_argument("-ext", help="sampling purpose: train OR test (used for file extension)", type=str, required=1)
    parser.add_argument("-ss_or_fr", help="use sample size or fraction of the data to sample: should be 'fr' or 'ss'", type=str, required=1)
    args = parser.parse_args()
    return args


def main():
    # -- Parse args -- #
    args = read_args()
    input_file = args.i
    dataset_size = int(args.dataset_size)
    data_fraction = args.fr
    sampling_seed = args.seed
    num_samples = args.num_s
    output_file_extension = args.ext
    sample_size = args.ss
    ss_or_fr = args.ss_or_fr

    # -- Define other params: output destination and filename(s), sampling probability etc. -- #
    input_file_portions = input_file.split("/")

    if ss_or_fr == 'fr':
        output_file_generic = input_file_portions[1] + '/' + input_file_portions[2] + '/Samples/' + input_file_portions[2].split(".")[0] + '__fr-' + str(data_fraction) + '__' # + str('1') + '.train/test'
        sampling_probability = 1 / (data_fraction * 1.0)
        sample_size = int(dataset_size / data_fraction)
    else:
        output_file_generic = input_file_portions[1] + '/' + input_file_portions[2] + '/Samples/' + input_file_portions[2].split(".")[0] + '__ss-' + str(sample_size) + '__' # + str('1') + '.train/test'
        sampling_probability = sample_size / (dataset_size * 1.0)

    print("sampling_prob")
    print(sampling_probability)

    # -- Seed -- #
    np.random.seed(sampling_seed)

    # -- Do the sampling -- #
    samples = [[] for _ in range(num_samples)]

    with open(input_file) as f:
        for dp_th, data_point in enumerate(f):
            # -- Each data point is or is not distributed to each sample depending on the probability corresponding to that specific sample -- #
            sample_probabilities_for_all_points = [np.random.random() for _ in range(num_samples)]
            for sth, sample in enumerate(samples):
                #print(sample_probabilities_for_all_points[sth], sampling_probability)
                if sample_probabilities_for_all_points[sth] < sampling_probability:
                    samples[sth].append(data_point)

        for sth, sample in enumerate(samples):
	    #print("going through samples to fill files")
            #print(len(sample))
	    # missing_data_points = sample_size - len(sample)
            # while missing_data_points > 0:
            #     with open(input_file) as f_bis:
            #         for dp_th, data_point in enumerate(f_bis):
            #             samples[sth].append(data_point)
            #             missing_data_points -= 1
            #
            # print(len(sample))
            with open(output_file_generic + str(sth + 1) + output_file_extension, 'w') as out_f:
                for dp_th, data_point in enumerate(sample):
                    out_f.write(data_point)






main()
