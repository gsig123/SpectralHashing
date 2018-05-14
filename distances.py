import numpy as np


def dist_euclidean(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def dist_hamming(x1, x2):
    xor = np.bitwise_xor(np.uint64(x1), np.uint64(x2))
    return np.count_nonzero(xor)

def dist_hamming_str(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# -- All credit for this function goes to https://github.com/wanji/sh/blob/master/sh.py -- #
def dist_hamming_between_datasets(B1, B2, BIT_CNT_MAP):
    if B1.ndim == 1:
        B1 = B1.reshape((1, -1))

    if B2.ndim == 1:
        B2 = B2.reshape((1, -1))

    npt1, dim1 = B1.shape
    npt2, dim2 = B2.shape

    if dim1 != dim2:
        raise Exception("Dimension not consists: %d, %d" % (dim1, dim2))

    Dh = np.zeros((npt1, npt2), np.uint16)

    for i in range(npt1):
        Dh[i, :] = BIT_CNT_MAP[np.bitwise_xor(B1[i, :], B2)].sum(1)

    return Dh