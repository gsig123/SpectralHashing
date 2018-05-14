from sklearn import manifold, datasets
import argparse
import numpy as np

# Usage: python "Dataset_Generators/generators__manifold-learning/swiss-roll.py" -n 1000 -noise 0.0 -seed 12 -output "Data/Manifold-Learning/swiss-roll_n-1000_noise-0.0.train"

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-output", help="swiss_roll_dataset output file", required=1)
    parser.add_argument("-n", help="n_samples", type=int, required=1)
    parser.add_argument("-seed", help="seed/random state", type=int, nargs='?', default=None)
    parser.add_argument("-noise", help="noise for generating swiss roll => in the interval [0.0, 1.0]", type=np.float, nargs='?', default=0.0)
    args = parser.parse_args()

    return args


def main_swiss_roll():
    args = read_args()

    n_samples = args.n
    swiss_roll_noise = args.noise
    output_file = args.output
    seed = args.seed
    dataset_swiss_roll, color = datasets.samples_generator.make_swiss_roll(n_samples, swiss_roll_noise, seed)

    color_col = np.array([[x] for x in color])
    dataset_swiss_roll_with_color_col = np.concatenate((dataset_swiss_roll, color_col), axis=1)

    np.savetxt(output_file, dataset_swiss_roll_with_color_col, delimiter=' ')


if __name__ == '__main__':
    main_swiss_roll()
