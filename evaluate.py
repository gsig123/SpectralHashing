import datetime
import numpy as np
import math
import random as rand
import argparse
from scipy.spatial import distance
from scipy.sparse import *
from scipy import *
from sklearn.neighbors import NearestNeighbors
from shparams import SHParam
from shparams import SHModel
from distances import dist_hamming
import pickle

def print_all_good_nn_indices_in_hamming(indices, hamm_dist_debug, text):
    counter = 0
    for ind, indices in enumerate(
            indices[hamm_dist_debug]):
        print("{0}th, #{1}: {2}".format(ind, len(indices), indices))
        counter += len(indices)
    print("\nFinal counter check for #{0} => {1}\n\n".format(text, counter))


# -- Evaluation 1: with Euclidean d_ball and hamm_d_ball in range(0, max_hamming_distance_tested) -- #
def evaluate_with_approximate_gt_d_balls(w_true_test_training, u_compactly_binarized_training, u_compactly_binarized_testing, u_training, u_testing,  max_hamming_distance_tested, BIT_CNT_MAP, eval_debug_object, debug_mode=False):
    total_good_pairs = w_true_test_training.sum()
    retrieved_good_pairs = np.zeros((max_hamming_distance_tested, 1))
    retrieved_pairs = np.zeros((max_hamming_distance_tested, 1))

    score_precision = np.zeros((max_hamming_distance_tested, 1))
    score_recall = np.zeros((max_hamming_distance_tested, 1))

    # print("score_recall", score_recall, max_hamming_distance_tested)
    # print("total_good_pairs", total_good_pairs)

    # for r, row in enumerate(w_true_test_training):
    #     print("{0} => {1}".format(r, sum(row)))

    for ith_testing, compact_testing_point in enumerate(u_compactly_binarized_testing):
        distances_from_testing_to_all_training = BIT_CNT_MAP[np.bitwise_xor(compact_testing_point, u_compactly_binarized_training)].sum(1)

        for hamming_distance_used_to_test_against in range(0, max_hamming_distance_tested):
            indices_pairs_of_good_pairs_in_d_hamm = np.where(distances_from_testing_to_all_training < hamming_distance_used_to_test_against + 0.00001)

            retrieved_good_pairs[hamming_distance_used_to_test_against][0] += sum(
                [w_true_test_training[ith_testing, pair_index] for pair_index in indices_pairs_of_good_pairs_in_d_hamm])
            retrieved_pairs[hamming_distance_used_to_test_against][0] += size(indices_pairs_of_good_pairs_in_d_hamm)

    for hamming_distance_used_to_test_against in range(0, max_hamming_distance_tested):
        score_precision[hamming_distance_used_to_test_against][0] = retrieved_good_pairs[hamming_distance_used_to_test_against][0] / (retrieved_pairs[hamming_distance_used_to_test_against][0] * 1.0 + 0.00000001)
        score_recall[hamming_distance_used_to_test_against][0] = retrieved_good_pairs[hamming_distance_used_to_test_against][0] / (total_good_pairs * 1.0 + 0.00000001)

    return score_precision.T, score_recall.T, eval_debug_object


def calculate_d_ball(average_number_neighbors, metric, data_norm):  # data_norm is usually data_train_norm in our case.
    nbrs = NearestNeighbors(n_neighbors=average_number_neighbors, algorithm='auto', metric=metric).fit(
        data_norm)
    distances, indices = nbrs.kneighbors(data_norm)
    # print("\n# -- Step calculate_d_ball, where distances are => -- #")
    # print(distances)
    d_ball = np.mean(distances[:, average_number_neighbors - 1])
    print("\n# -- d_ball = {0} -- #".format(d_ball))
    return d_ball

def calculate_approximate_ground_truth_with_d_ball(data_train_norm, data_test_norm, average_number_neighbors, n_test, n_train, number_of_splits_testing_gt):
    # -- Define d_ball from the training set -- #
    d_ball = calculate_d_ball(average_number_neighbors, 'euclidean', data_train_norm)
    print("\nPASSED d_ball calculation => dball = {0}\n".format(d_ball))


    # -- Define, for each testing point, which of the points in the training set is within its d_ball radius -- #
    w_true_test_training = csc_matrix((n_test, n_train), dtype=np.bool).todense()
    data_test_norm_chunks = np.array_split(data_test_norm, number_of_splits_testing_gt)
    to_index_to_store = 0
    for test_chunk_ith, test_chunk in enumerate(data_test_norm_chunks):
        d_true_test_training_chunk = distance.cdist(test_chunk, data_train_norm, metric='euclidean')
        from_index = to_index_to_store
        to_index = from_index + test_chunk.shape[0]
        to_index_to_store = to_index
        w_true_test_training[from_index:to_index, :] = d_true_test_training_chunk < d_ball


    return d_ball, w_true_test_training



# -- Evaluation 2: Exact GT, with reverse indices -- #
def get_reverse_indices(euclidean_indices, hamming_indices, k, query_size):
    all_queries_reverse_indices = np.zeros((query_size, k), dtype=int)
    for sq_ei_ith, single_query_euclidean_indices in enumerate(euclidean_indices):
        sq_reverse_indices = [hamming_indices[sq_ei_ith].tolist().index(eucl_index) for eucl_index_jth, eucl_index in
                              enumerate(single_query_euclidean_indices)]
        all_queries_reverse_indices[sq_ei_ith] = sq_reverse_indices
    return all_queries_reverse_indices


def evaluate_with_reverse_indices(data_train_norm, data_test_norm, u_training, u_testing, k):
    # Obs 1 => Euclidean space: query_set = data_test_norm, set_we_look_up = data_train_norm
    # Obs 2 => Hamming space: query_set = u_testing, set_we_look_up = u_training

    # -- Define vars -- #
    searching_space_size = data_train_norm.shape[0]
    query_size = data_test_norm.shape[0]
    num_of_batches = int(np.ceil(query_size / 10))

    # -- Split the query set -- #
    query_batches_hamming = array_split(u_testing, num_of_batches)
    query_batches_euclidean = array_split(data_test_norm, num_of_batches)

    # -- Find GT in Euclidean space -- #
    euclidean_nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(data_train_norm)
    # -- Find GT in Hamming space -- #
    hamming_nbrs = NearestNeighbors(n_neighbors=searching_space_size, algorithm='auto', metric='hamming').fit(u_training)

    # -- Make scale of retrieved samples needed for recall calculation -- #
    freqs = range(1, 1 + int(math.log2(searching_space_size)))  #  not so sure about this
    retrieved_points_scale = [np.power(2, x) for x in freqs]
    retrieved_points_scale.append(k)

    # -- Keep in mind that last row for results represents precision for the #bits we test and the provided k -- #
    results = np.zeros((len(retrieved_points_scale), 2))
    results[:, 0] = retrieved_points_scale

    for b_th, query_batch_euclidean in enumerate(query_batches_euclidean):
        euclidean_indices = euclidean_nbrs.kneighbors(query_batches_euclidean[b_th], return_distance=False)
        hamming_indices = hamming_nbrs.kneighbors(query_batches_hamming[b_th], return_distance=False)

        # -- Calculate reverse indices matrix -- #
        reverse_indices = get_reverse_indices(euclidean_indices, hamming_indices, k, len(query_batch_euclidean))

        for rp_th, retrieved_points in enumerate(retrieved_points_scale):
            # -- Calculate recall for this specific value of retrieved_points -- #
            true_positives = 0.0
            for qp_ri, query_point_reverse_indices in enumerate(reverse_indices):
                true_positives += sum((reverse_index < retrieved_points) for ri, reverse_index in enumerate(query_point_reverse_indices))

            # recall = true_positives / (query_size * 1.0 * k)
            results[rp_th, 1] += true_positives


    results[:, 1] /= query_size * 1.0 * k

    # Obs 3 => Last row in results corresponds to value for precision, for the selected #bits.
    return results