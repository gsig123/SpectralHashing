"""
Generator of 2d datasets with clusters + noise.
"""

import argparse
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def read_args():
    """Read arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-cluster",
                        help="""
                            A single cluster, consisting of an x and y coordinate and a radius.
                            Multiple clusters can be given.
                        """,
                        type=float, nargs=3,
                        action='append',
                        required=True)
    parser.add_argument("-noise",
                        type=float,
                        help="""
                            Probability a point is generated in a random location.
                            Should be a number between 0 and 1.
                        """,
                        required=True)
    parser.add_argument("-n", type=int, help="Number of data points", required=True)
    parser.add_argument("-output", help="Output file", required=True)
    return parser.parse_args()

def main():
    """Generate the data and plot it."""
    args = read_args()
    noise_prob = args.noise
    clusters = list(map(lambda r: {"x": r[0], "y": r[1], "radius": r[2]}, args.cluster))
    cluster_prob = (1.0-noise_prob)/len(clusters)

    points = []
    for _ in range(args.n):
        rand = random.random()
        if rand < noise_prob:
            points.append([random.random(), random.random()])
        else:
            cluster = clusters[int((rand-noise_prob)/cluster_prob)]
            distance = random.uniform(0, cluster["radius"])
            angle = random.uniform(0, 2*math.pi)
            points.append(
                [cluster["x"]+math.cos(angle)*distance, cluster["y"]+math.sin(angle)*distance])

    x_coords = list(map(lambda x: x[0], points))
    y_coords = list(map(lambda x: x[1], points))
    plt.plot(x_coords, y_coords, '.')
    plt.show()
    np.savetxt(args.output, points, delimiter=' ')

main()
