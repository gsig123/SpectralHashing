import argparse
import pickle
import numpy as np

from helpers import *


def read_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-input", help="input training set", required=1)
    parser.add_argument("-model", help="model output file", required=1)
    parser.add_argument("-bits", help="bits to encode to", type=int, required=1)
    parser.add_argument("-log", help="log file for training", type=str, required=1)
    args = parser.parse_args()

    return args


def train_sh_individually(input_file, output_model, bits_to_encode_to, log_file):
    print(input_file)
    # List of arrays of length 'dimension'
    data_train = np.genfromtxt(input_file, delimiter=' ', dtype=np.float)
    print("DONE reading training set")

    # List of arrays of length 'dimension', normalized to values between 0 and 1
    data_train_norm = normalize_data(data_train)
    print("DONE normalizing training set")

    sh_model = train_sh(data_train_norm, bits_to_encode_to, input_file, log_file)
    print("DONE model creation from training set")

    pickle.dump(sh_model, open(output_model, "wb"))
    print("DONE dumping model to file: {0}".format(output_model))


def main_train():
    # -- Read args -- #
    args = read_args()

    input_file = args.input
    output_model = args.model
    log_file = "log.log" if args.log is None else args.log
    train_sh_individually(args.input, args.model, args.bits, log_file)


main_train()

