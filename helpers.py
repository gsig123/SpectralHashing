import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import math
import json
import random as rand
import argparse
from scipy.spatial import distance
from scipy.sparse import *
from scipy import *
from sklearn.neighbors import NearestNeighbors
from sympy.combinatorics.graycode import GrayCode

from shparams import SHParam
from shparams import SHModel
from distances import *
from evaluate import *
import operator
import matplotlib.collections as collections

# -- 0. Algorithm-specific -- #
# data_norm: input data, list of normalized arrays of length 'dimension'
# data_train_filename_location: original input file, just for storing in model
def train_sh(data_norm, bits_to_encode_to, data_train_filename_location, log_file_train_destination):
    start = time.time()
    # -- Setup log file to save important info after training -- #
    with open(log_file_train_destination, 'w') as log_file_train:

        # -- Get training set dimensions -- #
        log_file_train.write("# -- Get training set dimensions -- #\n")
        # data_norm_n: number of training inputs
        # data_norm_d: number of dimensions
        data_norm_n, data_norm_d = data_norm.shape

        log_value_to_file(log_file_train, data_norm_n, "data_norm_n")
        log_value_to_file(log_file_train, data_norm_d, "data_norm_d")

        # -- PCA the data -- #
        log_file_train.write("# -- PCA the data -- #\n")
        n_pca = min(bits_to_encode_to, data_norm_d)
        cov_mat = np.cov(data_norm, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

        log_array_to_file(log_file_train, cov_mat, "cov_mat")
        log_array_to_file(log_file_train, eig_vals, 'eigen values')
        log_array_to_file(log_file_train, eig_vecs, 'eigen vectors')
        # -- Do eigenvectors check -- #
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        print('Everything ok, eigenvectors have all the same unit length 1!')

        # NB: signs of pc are slightly different than in matlab!!!! => same for the signs in data_norm_pcaed

        pc = np.fliplr(eig_vecs[:, -n_pca:])  # flip array in left-right direction
        # input data, mapped to the principal component axes.
        data_norm_pcaed = data_norm.dot(pc)
        log_array_to_file(log_file_train, pc, 'principal components')

        # -- Fit uniform distribution -- #
        log_headline(log_file_train, "# -- Fit uniform distribution -- #")
        eps = np.finfo(float).eps
        mn = data_norm_pcaed.min(axis=0) - eps
        mx = data_norm_pcaed.max(axis=0) + eps
        log_value_to_file(log_file_train, bits_to_encode_to, "bits_to_encode_to")
        log_value_to_file(log_file_train, mn, "mn")
        log_value_to_file(log_file_train, mx, "mx")

        # -- Enumerate eigenfunctions -- #
        log_headline(log_file_train, "# -- Enumerate eigenfunctions -- #")
        r = (mx - mn)
        norm_r = r/max(r)  #  + 0.000001
        max_mode = np.ceil((bits_to_encode_to + 1) * norm_r).astype(int)
        n_modes = np.sum(max_mode) - n_pca + 1
        log_value_to_file(log_file_train, max_mode, "max_mode")
        log_value_to_file(log_file_train, n_modes, "n_modes")
        log_value_to_file(log_file_train, r, "r")
        log_value_to_file(log_file_train, norm_r, "norm_r")

        # fill modes with the different possible modes for each pca.
        # index 0 is ignored
        m = 0
        modes = np.ones((int(n_modes), n_pca))
        for ith_pc in range(0, n_pca):
            row_index_to = m + max_mode[ith_pc]
            modes[(m + 1):row_index_to, ith_pc] = np.arange(2, max_mode[ith_pc] + 1)
            m = row_index_to - 1
        modes -= 1

        # log_array_to_file(log_file_train, modes, "modes")
        omega_zero = math.pi / r #  + 0.000001
        # log_value_to_file(log_file_train, omega_zero, "omega_zero")
        omegas = modes * np.tile(omega_zero, (n_modes, 1))
        # log_array_to_file(log_file_train, omegas, "omegas")

        eig_val = np.sum(np.power(omegas, 2), axis=1)
        log_array_to_file(log_file_train, sort(eig_val), "eig_val, but sorted")
        # eig_val indexes sorted by least first.
        ii = eig_val.argsort()
        modes = modes[ii[1:bits_to_encode_to + 1], :]
        log_array_to_file(log_file_train, modes, "final modes")


        # -- Store how many times each PC gets cut (basically how many bits each PC gets) -- #
        store_num_of_bits_each_pc_gets(modes, log_file_train)

        # sh_param = SHParam(mn, mx, modes, bits_to_encode_to)
        sh_model = SHModel(mn, mx, modes, bits_to_encode_to, pc, omega_zero, data_train_filename_location)

        # -- Time training phase -- #
        elapsed_time_formatted = time_process(start, time.time())
        log_value_to_file(log_file_train, elapsed_time_formatted, "elapsed time for training")

    # -- Close log training file -- #
    log_file_train.close()

    return sh_model



def plot_sine_partitioning_vanilla_sh(data_norm_pcaed_and_centered, omega_i, ys):
    data_box = data_norm_pcaed_and_centered * omega_i + math.pi / 2
    nonzero_col_index = np.nonzero(data_norm_pcaed_and_centered * omega_i)[1][0]
    x_lin_data = data_box[:, nonzero_col_index]
    y_sin_data = ys[:, nonzero_col_index]

    x_y = zip(x_lin_data, y_sin_data)
    x_y_sorted = sorted(list(x_y).copy(), key=lambda x: x[0])
    x, y = zip(*x_y_sorted)

    plot_sin(x, y)
    intersection_indices_for_ith_bit = np.argwhere(np.diff(np.sign(y)) != 0).reshape(-1) + 0
    print("intersection_indices_for_ith_bit", intersection_indices_for_ith_bit)

    for ith, index in enumerate(intersection_indices_for_ith_bit):
        # -- Obs(***): The visualization for vertical cuts will be more precise the more points we have to approximate the sine curve. Otherwise, it will show a hard cut, not excatly intersecting the x-axis -- #
        plt.axvline(x=x[index],
                    color='black',
                    linewidth=1,
                    linestyle='--')
    plot_x_axis_data(data_box[:, nonzero_col_index], labeling=True)
    plt.show()

def get_sine_data_box(data_norm_pcaed_and_centered, sh_model, data_norm_n):
    omega_zero_non_pi = sh_model.omega_zero #/ math.pi

    omegas_compress_training = sh_model.modes * np.tile(omega_zero_non_pi, (sh_model.n_bits, 1))
    data_box = np.zeros((data_norm_n, sh_model.n_bits))

    for ith_bit in range(0, sh_model.n_bits):
        omega_i = np.tile(omegas_compress_training[ith_bit, :], (data_norm_n, 1))
        nonzero_col_index = np.nonzero(np.all(omega_i != 0, axis=0))[0][0]
        data_box[:, ith_bit] = data_norm_pcaed_and_centered[:, nonzero_col_index] * omega_i[:, nonzero_col_index] + math.pi / 2
    return data_box


def print_min_max_data_box(data_box, label):
    max_column_data_box = data_box.max(axis=0)
    min_column_data_box = data_box.min(axis=0)


def get_pc_bitwise_contribution(sh_model):
    bits_per_pcs = {}
    for ith_pc, pc_mode_cut in enumerate(sh_model.modes.T):
        cuts_for_ith_pc = [item for item in pc_mode_cut if item != 0]
        if len(cuts_for_ith_pc) > 0:
            bits_per_pcs[str(ith_pc)] = cuts_for_ith_pc
    return bits_per_pcs


def get_pc_order_as_vanilla(sh_model):
    unordered_modes = sh_model.modes
    ordered_modes = order_modes_for_bit_contribution(sh_model)
    pcs_ordered = []

    for ord_mode in ordered_modes:
        nonzero_col_index = np.nonzero(ord_mode)[0][0]
        if not nonzero_col_index in pcs_ordered:
            pcs_ordered.append(nonzero_col_index)

    return pcs_ordered


def get_next_pc_to_attach_bits_from(bits_per_pcs, pcs_ordered, pc):
    pcs_to_sum = [0]
    pcs_to_sum.extend(
        [len(bits_per_pcs[str(pcth)]) for pcth in pcs_ordered if str(pcth) in bits_per_pcs.keys() and pcth < pc])
    pc = sum(pcs_to_sum)
    return pc

def order_modes_for_bit_contribution(sh_model):
    ordered_modes = np.zeros(sh_model.modes.shape)
    filled_row_th = 0
    for ith_pc, pc_mode_cut in enumerate(sh_model.modes.T):
        pc_mode_cut_indices = [i for i, item in enumerate(pc_mode_cut) if item != 0]
        if max(pc_mode_cut) == min(pc_mode_cut) and max(pc_mode_cut) == 0:
            print("The {0}th mode is all-zeros and doesn't contribute!!!".format(ith_pc))
        for cut_index in pc_mode_cut_indices:
            # print("cut_index: {0}".format(cut_index))
            ordered_modes[filled_row_th, :] = sh_model.modes[cut_index, :]
            filled_row_th += 1

    return ordered_modes



def get_data_box_info(data_box_train, pc, num_buckets_per_pc):
    pc_scores_train = data_box_train[:, pc]
    max_pc_score_train = max(pc_scores_train)
    min_pc_score_train = min(pc_scores_train)
    range_pc_train = max_pc_score_train - min_pc_score_train
    interval_pc_train = range_pc_train / (num_buckets_per_pc * 1.0)
    return pc_scores_train, max_pc_score_train, min_pc_score_train, range_pc_train, interval_pc_train



def get_provenance_bucket_index(pc_score, min_pc_score_train, interval_pc_train, num_gray_codes):
    bucket_index = int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))
    if bucket_index < 0:
        bucket_index = 0
    provenance_bucket_index = bucket_index if bucket_index < num_gray_codes else num_gray_codes - 1
    return provenance_bucket_index

def get_pcs_ith_bits_mapping(sh_model):
    pcs_ith_bits_mapping = {}
    pcs_ith_bits_when_multiple_cuts = []
    first_pcs_when_axis_cut_multiple_times =[]
    for ith_pc, pc_mode_cut in enumerate(sh_model.modes.T):
        cuts_for_ith_pc = sorted([i for i, item in enumerate(pc_mode_cut) if item != 0])
        cuts_for_ith_pc_indices = [(index, val) for index, val in enumerate(cuts_for_ith_pc)]
        cuts_for_ith_pc_len = len(cuts_for_ith_pc)
        if cuts_for_ith_pc_len > 0:
            pcs_ith_bits_mapping[str(ith_pc)] = cuts_for_ith_pc
            if cuts_for_ith_pc_len > 1:
                pcs_ith_bits_when_multiple_cuts.extend(cuts_for_ith_pc_indices[1:])
                first_pcs_when_axis_cut_multiple_times.append(cuts_for_ith_pc_indices[0][1])
        else:
            pcs_ith_bits_mapping[str(ith_pc)] = []

    return pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times







def get_pc_bitwise_contribution_and_ordering(sh_model):
    bits_contribution_and_ordering_per_pcs = {}
    for ith_pc, pc_mode_cut in enumerate(sh_model.modes.T):
        cuts_for_ith_pc = [i for i, item in enumerate(pc_mode_cut) if item != 0]
        if len(cuts_for_ith_pc) > 0:
            bits_contribution_and_ordering_per_pcs[str(ith_pc)] = cuts_for_ith_pc

    sorted_dict = sorted(bits_contribution_and_ordering_per_pcs.items(), key=operator.itemgetter(1))

    ordered_keys = []
    for tup in sorted_dict:
        ordered_keys.append(int(tup[0]))

    return ordered_keys, bits_contribution_and_ordering_per_pcs


def find_median_recursively__short(data, times):
    if times == 1:
        return [np.median(data)]
    else:
        interval = 100.0 / (times + 1)
        percentiles = [interval * pth for pth in range(1, times + 1)]
        medians = [np.percentile(data, int(percentile)) for percentile in percentiles]
        return medians

def find_median_recursively__long(data, times):
    list_length = len(data)
    if times == 1:
        return np.median(data)
    else:
        median = np.median(data)

        if times % 2 == 1:
            times_left = (times - 1) / 2
            data = sorted(data)
            if list_length % 2 == 1:
                median_index = np.where(data == median)[0][0]
                left_list = data[:median_index]
                right_list = data[median_index + 1:]
            else:
                left_list = data[:int(list_length / 2)]
                right_list = data[int(list_length / 2):]

            return [median, find_median_recursively__long(left_list, times_left), find_median_recursively__long(right_list, times_left)]
        else:
            interval = 100.0 / (times + 1)
            percentiles = [interval * pth for pth in range(1, times + 1)]
            medians = [np.percentile(data, int(percentile)) for percentile in percentiles]
            return medians





# -- I. Normalization -- #
def normalize_data(data):
    data_norm = []
    for dimension in data.T:
        min_val_dim = min(dimension)
        max_val_dim = max(dimension)
        data_norm.append([(value - min_val_dim) / (max_val_dim - min_val_dim + 0.000001) for value in dimension]) #  added the 0.00001 because otherwise I divide the values by 0 and that gives nan in the covariance matrix!!!! (so we avoid the edge case)
    return np.array(data_norm).T


# -- II. Evaluation Formulas -- #
def calculate_f_score(precision_score, recall_score):
    return (2.0 * precision_score * recall_score) / (precision_score + recall_score + 0.000001)


# -- III. Printing -- #
def print_scores(score_precision, score_recall, score_f_measure):
    print(score_precision)
    print(score_recall)
    print(score_f_measure)

def print_help(text, var):
    print(text + " =>")
    print(var)


def print_compressed_dataset_hashcodes(dataset):
    u_training_int = np.array(dataset, dtype=int)
    u_training_int_hashcodes = [''.join(str(bit) for bit in binary_vector) for binary_vector in u_training_int]
    for hc, hashcode in enumerate(u_training_int_hashcodes):
        print("{0}th: {1}".format(hc, hashcode))



# -- IV. Different checks -- #
def calculate_hamming_dist_between_all_neighboring_buckets(all_unique_hashcodes):
    if len(all_unique_hashcodes) > 1:
        dist_hamming_between_neighboring_buckets = [dist_hamming_str(all_unique_hashcodes[b_hc - 1], bucket_hashcode) for b_hc, bucket_hashcode in enumerate(all_unique_hashcodes)][1:]
        print("Hamming distances between all neighboring non-empty buckets are => ")
        print(dist_hamming_between_neighboring_buckets)
        labels = [str(dh) + ": " + str(dist_hamm) for dh, dist_hamm in enumerate(dist_hamming_between_neighboring_buckets)]
        plot_bar_chart(dist_hamming_between_neighboring_buckets, labels, 0.75, "Hamming distances between all neighboring buckets", max(dist_hamming_between_neighboring_buckets) + 1)
    else:
        print("Everything got hashed to the same bucket!!!!")



def check_data_distribution_into_buckets(n_bits, u_training, log_file):
    np.set_printoptions(threshold=np.nan)

    u_training_int = np.array(u_training, dtype=int)
    u_training_int_hashcodes = [''.join(str(bit) for bit in binary_vector) for binary_vector in u_training_int]

    # -- Calculate buckets' hashcodes in their order of appearance -- #
    all_unique_hashcodes = []
    for hash_code in u_training_int_hashcodes:
        if hash_code not in all_unique_hashcodes:
            all_unique_hashcodes.append(hash_code)

    # print("all_unique_hashcodes in order of appearance =>")
    # print(all_unique_hashcodes)

    # -- Calculate occurrences for each bucket hashcode, again, in the order of appearance -- #
    unique_hashcodes_occurrences = [(unique_hashcode, u_training_int_hashcodes.count(unique_hashcode)) for unique_hashcode in all_unique_hashcodes]
    # print("unique_hashcodes_occurrences")
    # print(unique_hashcodes_occurrences)
    labels = [str(p) + " - " + pair[0] + ", " + "{0:.1f}".format(pair[1]/u_training.shape[0] * 100.0) for p, pair in enumerate(unique_hashcodes_occurrences)]
    plot_bar_chart([pair[1]/u_training.shape[0] * 100.0 for pair in unique_hashcodes_occurrences], labels, 0.75, "How data 'falls' into buckets")

    return all_unique_hashcodes






def check_buckets_balance_constraint(n_bits, u_compactly_binarized_set, log_file):  # usually, in this setup, we pass the training set
    np.set_printoptions(threshold=np.nan)

    u_compactly_binarized_set_row_concatenated = [''.join(str(x) for x in row) for row in u_compactly_binarized_set]
    unique, counts = np.unique(u_compactly_binarized_set_row_concatenated, return_counts=True)

    if np.log2(u_compactly_binarized_set.shape[0]) < n_bits or n_bits > 62:  # Case 1 (many buckets, few data points): we have way many more buckets than data points > we expect at most 1 elem in each bucket
        avg = 1.0
        c = len(counts)
        case = "buckets >> data points: we expect avg to be ~ 1.0"
    else:  # Case 2 (few buckets, many data points): we expect the stddev of counts (with respect to the total number of buckets) to be ~0.
        n_buckets = np.power(2, n_bits)
        avg = sum(counts) / (n_buckets * 1.0)  # unbiased avg
        c = n_buckets
        case = "buckets << data points: we expect stddev to be ~ 0.0"

    stddev = np.sqrt(sum([np.power(count - avg, 2) for count in counts]) / c)

    max_bucket_size = max(counts)
    min_bucket_size = min(counts)

    dataset_size = u_compactly_binarized_set.shape[0]
    counts_percentage = ["{0:.3f}".format(b/dataset_size) for b in counts]
    buckets_magnitude = [int(b/min_bucket_size) for b in counts]

    with open(log_file, 'a') as log_f:
        print("\n\n\n# -- START: Check buckets' balance constraint -- #")
        log_f.write("\n\n\n# -- START: Check buckets' balance constraint -- #\n\n\n")

        print("\t* n_bits => {}".format(n_bits))
        log_value_to_file(log_f, n_bits, "n_bits")

        print("\t* n_non-empty_buckets => {}".format(len(counts)))
        log_value_to_file(log_f, len(counts), "n_non-empty_buckets")


        print("\t* case => {}".format(case))
        log_value_to_file(log_f, case, "case")

        print("\t* avg => {}".format(avg))
        log_value_to_file(log_f, avg, "avg")

        print("\t* stddev => {}".format(stddev))
        log_value_to_file(log_f, stddev, "stddev")

        print("\t* smallest bucket size => {}".format(min_bucket_size))
        log_value_to_file(log_f, min_bucket_size, "min_bucket_size")

        print("\t* biggest bucket size => {}".format(max_bucket_size))
        log_value_to_file(log_f, max_bucket_size, "max_bucket_size")

        print("\t* hashed dataset size => {}".format(dataset_size))
        log_value_to_file(log_f, dataset_size, "dataset_size")

        print("\t* repartition diff between smallest and biggest buckets => {}".format((max_bucket_size - min_bucket_size)))
        log_value_to_file(log_f, (max_bucket_size - min_bucket_size), "repartition diff between smallest and biggest buckets")

        print("\t* buckets' sizes are => {}".format(counts))
        log_value_to_file(log_f, np.array(counts), "buckets' sizes")

        print("\t* how many times each bucket is bigger than the smallest bucket => {}".format(buckets_magnitude))
        log_value_to_file(log_f, np.array(buckets_magnitude), "how many times each bucket is bigger than the smallest bucket")

        print("\t* buckets' sizes in % are => {}".format(counts_percentage))
        log_value_to_file(log_f, np.array(counts_percentage), "buckets' sizes in %")
        print("# -- END: Check buckets' balance constraint -- #")

    # -- Close others log file -- #
    log_f.close()


def store_num_of_bits_each_pc_gets(modes, log_file):
    num_of_bits_each_pc_gets = {}
    for ith_pc, pc_mode_cut in enumerate(modes.T):
        cuts_for_ith_pc = [item for item in pc_mode_cut if item != 0]
        if len(cuts_for_ith_pc) > 0:
            num_of_bits_each_pc_gets['PC_' + str(ith_pc)] = cuts_for_ith_pc
    log_dict_to_file(log_file, num_of_bits_each_pc_gets,
                     "num_of_bits_each_pc_gets (cuts and each cut's frequency)")




# -- V. Binary vectors compaction -- #
# -- Bit map for fast binary calculations -- #
def init_bitmap():
    BIT_CNT_MAP = np.array([bin(i).count("1") for i in range(256)], np.uint16)
    return BIT_CNT_MAP

# -- All credit for this function goes to https://github.com/wanji/sh/blob/master/sh.py -- #
def compact_bit_matrix(bit_matrix):
    n_size, n_bits = bit_matrix.shape
    n_words = (n_bits + 7) / 8
    b = np.hstack([np.packbits(bit_matrix[:, i*8:(i+1)*8][:, ::-1], 1)
                   for i in range(int(n_words))])
    residue = n_bits % 8
    if residue != 0:
        b[:, -1] = np.right_shift(b[:, -1], 8 - residue)

    return b


def compact_bit_matrix_v2(bit_matrix):
    compact_bit_matrix = 0
    for bit_vector_row in bit_matrix:
        compact_bit_matrix = (compact_bit_matrix << 1) | bit_vector_row
    return compact_bit_matrix




# -- VI. Timing -- #
def time_process(start, end):
    elapsed_time = end - start
    elapsed_time_formatted = "{0}:{1} => min:sec".format(time.localtime(elapsed_time).tm_min, time.localtime(elapsed_time).tm_sec)
    return elapsed_time_formatted


# -- VII. Logging -- #
def log_array_to_file(log_file, array_to_log, comment_str):
    log_file.write(comment_str + " => \n{0}\n\n".format("\n".join(map(str, array_to_log))))

def log_dict_to_file(log_file, dict_to_log, comment_str):
    log_file.write(comment_str + " => \n")
    for id, values in dict_to_log.items():
        log_file.write(':'.join([id]) + " : " + str(values) + '\n')
    log_file.write("\n\n")

def log_value_to_file(log_file, value, comment_str):
    log_file.write(comment_str + " => \n{0}\n\n".format(value))

def log_headline(log_file, headline):
    log_file.write("\n\n\n\n" + headline + "\n")


# -- VIII. Plotting -- #
def plot_bar_chart(data, labels, bar_width, title, max_y=100, bar_color="teal"):
    num_items = len(data)
    bar_chart = plt.figure(figsize=(19, 3))

    ind = np.arange(num_items)
    plt.title(title)
    plt.bar(ind, data, width=bar_width, color=bar_color)
    plt.xticks(ind + bar_width / 2, labels)
    bar_chart.autofmt_xdate()
    plt.ylim([0, max_y])
    plt.show()


def plot_sin(x_lin_data, y_sin_data, color="#808080"):
    y_sin_data_positive = [y_sin_data_row for i, y_sin_data_row in enumerate(y_sin_data) if y_sin_data_row > 0]
    x_sin_data_positive = [x_lin_data[i] for i, y_sin_data_row in enumerate(y_sin_data) if y_sin_data_row > 0]
    y_sin_data_negative = [y_sin_data_row for i, y_sin_data_row in enumerate(y_sin_data) if y_sin_data_row <= 0]
    x_sin_data_negative = [x_lin_data[i] for i, y_sin_data_row in enumerate(y_sin_data) if y_sin_data_row <= 0]

    plt.xlim([min(x_lin_data), max(x_lin_data)])
    plt.ylim([-1, 1])

    plt.axhline(y=0.0, color='#efefef', linestyle='-')
    plt.plot(x_sin_data_positive, y_sin_data_positive, color='g', linewidth=2)
    plt.plot(x_sin_data_negative, y_sin_data_negative, color='r', linewidth=2)




def plot_x_axis_data(x_axis_data, color="black", labeling=True, debug_label_value=[]): # [47]

    plt.xlim([min(x_axis_data), max(x_axis_data)])
    plt.ylim([-1, 1])
    plt.scatter(x_axis_data, np.zeros(len(x_axis_data)), color=color, s=2, zorder=1000)
    plt.xlabel("PC scores")
    plt.ylabel("Eigenfunction values for PC scores")
    if labeling:
        for label, score_point in enumerate(x_axis_data):
            if label in debug_label_value:
                plt.annotate(label, (x_axis_data[label], 0))


def plot_2D(dataset_1, show_cut_points, headline, cut_points_pc=[1, 2, 3], color="black", debug_label_value=[]): # [47]
    min_x_dim = min(dataset_1[:, 0])
    max_x_dim = max(dataset_1[:, 0])
    min_y_dim = min(dataset_1[:, 1])
    max_y_dim = max(dataset_1[:, 1])

    fig = plt.figure(figsize=(5, 5))
    point_size = 2

    # ---- First subplot
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([min_x_dim, max_x_dim])
    ax.set_ylim([min_y_dim, max_y_dim])

    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        c="black",
        s=point_size)

    for label, score_point in enumerate(dataset_1[:, 0]):
        if label in debug_label_value:
            plt.annotate(label, (dataset_1[label, 0], dataset_1[label, 1]))

    if show_cut_points:
        print("cut_points_pc: {0}".format(cut_points_pc))
        # plt.axhline(y=0.0, color='#efefef', linestyle='-')
        for cth, cp in enumerate(cut_points_pc):
            # print("index", cp)
            plt.axvline(x=cp,
                        color=color,
                        linewidth=1)


    plt.title("Dataset: {0}".format(headline))
    plt.show()
