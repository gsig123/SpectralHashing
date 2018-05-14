from sklearn import manifold, datasets
import argparse
import numpy as np

# Usage: python "Dataset_Generators/generators__manifold-learning/s-curve.py" -n 1000 -noise 0.0 -seed 12 -output "Data/Manifold-Learning/s-curve_n-1000_noise-0.0.train"

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-output", help="s_curve_dataset output file", required=1)
    parser.add_argument("-n", help="n_samples", type=int, required=1)
    parser.add_argument("-seed", help="seed/random state", type=int, nargs='?', default=None)
    parser.add_argument("-noise", help="noise for generating s curve => in the interval [0.0, 1.0]", type=np.float, nargs='?', default=0.0)
    args = parser.parse_args()

    return args


def main_s_curve():
    args = read_args()

    n_samples = args.n
    s_curve_noise = args.noise
    output_file = args.output
    seed = args.seed
    dataset_s_curve, color = datasets.samples_generator.make_s_curve(n_samples, s_curve_noise, seed)

    color_col = np.array([[x] for x in color])
    dataset_s_curve_with_color_col = np.concatenate((dataset_s_curve, color_col), axis=1)

    np.savetxt(output_file, dataset_s_curve_with_color_col, delimiter=' ')


if __name__ == '__main__':
    main_s_curve()
