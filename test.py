import numpy as np
import pickle
import argparse
import os.path
import importlib

from sklearn.neighbors.dist_metrics import DistanceMetric

from helpers import *
from shparams import ApproxGT
from distances import *
from classes import EvalDebugApproachComponents


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="model file to test on", required=1)
    parser.add_argument("-test_file", help="input testing file", required=1)
    parser.add_argument("-log_file_test", help="log file for testing", type=str, required=1)
    parser.add_argument("-compressor", help="A module containing a compress(data_norm, sh_model, label) function", default='Compressors.vanilla')
    parser.add_argument("-log_file_others", help="log file for other aspects", type=str, required=1)

    parser.add_argument("-k", help="average number of nearest neighbours", type=int, default=10)
    parser.add_argument("-nost", help="number of splits/batches for testing set when calculating the ground truth", type=int, nargs='?', default=10)
    parser.add_argument("-mhd", help="maximum Hamming distance used for testing(in evaluation)", type=int, default=30)
    parser.add_argument("-eval_type", help="type of evaluation: approx GT 1a || approx GT 1b || precise GT with reverse indices", type=int, default=0)
    parser.add_argument("-ordered_pcs", help="should the pcs/bits' attachment be ordered or not?: uord || ord", type=str, default='ord')
    args = parser.parse_args()

    return args


def get_and_print_buckets(unq_indices_training, unique_buckets_and_indices_training, u_training_int_hashcodes):
    for unq_i in unq_indices_training:
        unique_buckets_and_indices_training[u_training_int_hashcodes[unq_i[0]]] = unq_i
    for k, val in unique_buckets_and_indices_training.items():
        print("bucket: " + k + ", with {0} points' indices: {1}".format(len(val), val))
    return unique_buckets_and_indices_training


def evaluate_approximate_gt(
                            data_train_norm,
                            data_test_norm,
                            average_number_neighbors,
                            n_test,
                            n_train,
                            number_of_splits_testing_gt,
                            u_compactly_binarized_training,
                            u_compactly_binarized_testing,
                            u_training,
                            u_testing,
                            max_hamming_distance_tested,
                            sh_model_training_filename,
                            compress_type,
                            ordered_pcs,
                            debug_mode):

    approx_gt_filename = "Results/" + sh_model_training_filename + "approx.gt"

    # -- Calculate gt model from file -- #
    d_ball_eucl, w_true_test_training = calculate_approximate_ground_truth_with_d_ball(
                                            data_train_norm,
                                            data_test_norm,
                                            average_number_neighbors,
                                            n_test,
                                            n_train,
                                            number_of_splits_testing_gt)

        # -- Store approximate GT (ground truth) in file -- #
    #     approx_gt = ApproxGT(w_true_test_training, d_ball_eucl, average_number_neighbors, n_train, n_test,approx_gt_filename)
    # pickle.dump(approx_gt, open(approx_gt_filename, "wb"))

    # -- Announce if GT matrix w_true_test_training doesn't even have nn, for any query point in the testing set -- #
    if debug_mode and not any(w_true_test_training):
        print("# -- NO good NN at all for the given query/testing set, as w_true_test_training is all with 0's!!!! -- #\n")

    # -- Evaluate precision && recall for the inputted value of bits_to_encode category -- #
    BIT_CNT_MAP = init_bitmap()


    score_recall = np.zeros((max_hamming_distance_tested, 1))
    score_precision = np.zeros((max_hamming_distance_tested, 1))
    # -- Evaluate approximate GT precision && recall with d_ball(s) for the given bits_to_encode category -- #

    # if debug_mode:
    # -- Find out how many nn we found on avg in the Euclidean hamm_ball we calculated -- #
    np.set_printoptions(threshold='nan')
    np.set_printoptions(threshold=np.nan)
    print("# -- START: Common sense check of avg nns in d_ball -- #")
    print("w_true_test_training => \n")
    stddev_nn_in_d_ball = np.std([sum(row) for row in w_true_test_training])
    print("\nThe # good nns found in the Euclidean d_ball are {0}, and they should be close to k={1}, while the stddev for each query point's nn is {2}\n".format(np.sum(w_true_test_training) / w_true_test_training.shape[0], average_number_neighbors, stddev_nn_in_d_ball))
    print("# -- END: Common sense check of avg nns in d_ball -- # \n")

    # -- Init Debug object for respective approach (vanilla, balanced or median) -- #
    eval_debug_object = EvalDebugApproachComponents()


    score_precision[:, 0], score_recall[:, 0], eval_debug_object = evaluate_with_approximate_gt_d_balls(
        w_true_test_training,
        u_compactly_binarized_training,
        u_compactly_binarized_testing,
        u_training,
        u_testing,
        max_hamming_distance_tested,
        BIT_CNT_MAP,
        eval_debug_object,
        debug_mode)
    score_f_measure = calculate_f_score(score_precision, score_recall)

    if debug_mode:
        # -- Print u_training and u_testing and see how buckets are formed for u_training -- #
        unique_buckets_and_indices_training = {}
        unique_buckets_and_indices_testing = {}
        u_training_int = np.array(u_training, dtype=int)
        u_testing_int = np.array(u_testing, dtype=int)
        u_training_int_hashcodes = [''.join(str(bit) for bit in binary_vector) for binary_vector in u_training_int]
        u_testing_int_hashcodes = [''.join(str(bit) for bit in binary_vector) for binary_vector in u_testing_int]
        unq_training, unq_inv_training, unq_cnt_training = np.unique(u_training_int_hashcodes, return_inverse=True, return_counts=True)
        unq_testing, unq_inv_testing, unq_cnt_testing = np.unique(u_testing_int_hashcodes, return_inverse=True, return_counts=True)
        unq_indices_training = np.split(np.argsort(unq_inv_training), np.cumsum(unq_cnt_training[:-1]))
        unq_indices_testing = np.split(np.argsort(unq_inv_testing), np.cumsum(unq_cnt_testing[:-1]))
        unique_buckets_and_indices_training = get_and_print_buckets(unq_indices_training, unique_buckets_and_indices_training, u_training_int_hashcodes)
        unique_buckets_and_indices_testing = get_and_print_buckets(unq_indices_testing, unique_buckets_and_indices_testing, u_testing_int_hashcodes)

        eval_debug_object.__set_compress_type__(compress_type)
        eval_debug_object.__set_unique_buckets_and_indices__(unique_buckets_and_indices_training, unique_buckets_and_indices_testing)
        eval_debug_object.__set_u_training_and_u_testing__([''.join(str(bit) for bit in binary_vector) for binary_vector in np.array(u_training, dtype=int)], [''.join(str(bit) for bit in binary_vector) for binary_vector in np.array(u_testing, dtype=int)])

        # 4. Store all eval debug info for the respective approach (vanilla, balanced or median) so comparison is made afterwards
        eval_filename ="Results/" + sh_model_training_filename.split("/")[2] + "/Logs/" + \
                         sh_model_training_filename.split("/")[-1] + ".eval.debug." + compress_type
        pickle.dump(eval_debug_object, open(eval_filename, "wb"))


    return score_precision, score_recall, score_f_measure


def log_evaluation_results(log_file_test_destination, metrics_eval, num_rows):
    # -- Setup log file to save quality measures after testing -- #
    with open(log_file_test_destination, 'w') as log_file_test:
        for row_index in range(0, num_rows):
            log_file_test.write(
                "\t".join(map(str, [format(score, '.10f') for score in metrics_eval[row_index]])) + "\n")
    # -- Close log testing file -- #
    log_file_test.close()


def main_test():
    args = read_args()
    sh_model = pickle.load(open(args.model, "rb"))
    print("DONE reading model from file")

    # -- Print modes -- #
    print_help("Modes from training", sh_model.modes)

    # -- Define params && arguments -- #
    training_filename = sh_model.training_filename
    average_number_neighbors = args.k
    number_of_splits_testing_gt = args.nost
    max_hamming_distance_tested = args.mhd
    log_file_test_destination = args.log_file_test
    log_others = args.log_file_others
    eval_type = args.eval_type
    compress_type = args.compressor
    ordered_pcs = args.ordered_pcs
    compressor = importlib.import_module(args.compressor)

    # -- Import datasets -- #
    data_train = np.genfromtxt(training_filename, delimiter=' ', dtype=np.float)
    data_test = np.genfromtxt(args.test_file, delimiter=' ', dtype=np.float)
    print("DONE reading training && testing set")

    # -- Normalize datasets -- #
    data_train_norm = normalize_data(data_train)
    data_test_norm = normalize_data(data_test)
    print("DONE normalizing training && testing set")

    # -- Get datasets' sizes -- #
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]

    # -- compressSH.m: For training set -- #
    u_training, u_compactly_binarized_training = compressor.compress(data_train_norm, sh_model, "training")
    print("DONE compressing training set\n")
    # -- compressSH.m: For testing set -- #
    u_testing, u_compactly_binarized_testing = compressor.compress(data_test_norm, sh_model, "testing")
    print("DONE compressing testing set\n")

    start = time.time()

    if eval_type == 0:
        # -- *** Evaluation, Approach 1a: with an Euclidean d_ball and a Hamming d_ball in a given range(0, max_hamming_distance_tested), where max is manually chosen -- #
        # Prepare params to create ApproxGT model and store it in a file
        score_precision, score_recall, score_f_measure = evaluate_approximate_gt(
                                                                data_train_norm,
                                                                data_test_norm,
                                                                average_number_neighbors,
                                                                n_test,
                                                                n_train,
                                                                number_of_splits_testing_gt,
                                                                u_compactly_binarized_training,
                                                                u_compactly_binarized_testing,
                                                                u_training,
                                                                u_testing,
                                                                max_hamming_distance_tested,
                                                                sh_model.training_filename,
                                                                compressor,
                                                                ordered_pcs,
                                                                False)
        log_file_test_destination += ".0"

        print("\nscore_precision")
        print(score_precision)
        print("score_recall")
        print(score_recall)
        # print("score_f_measure")
        # print(score_f_measure)

    elif eval_type == 1:
        # -- *** Evaluation, Approach 1b: with an Euclidean d_ball and an avg Hamming d_ball, which is calculated same as the Euclidean d_ball, but in Hamming space -- #
        d_hamm_ball = calculate_d_ball(average_number_neighbors, dist_hamming, u_training)
        max_hamming_distance_tested = int(np.ceil(d_hamm_ball)) + 20

        score_precision, score_recall, score_f_measure = evaluate_approximate_gt(
                                                                data_train_norm, data_test_norm,
                                                                average_number_neighbors,
                                                                n_test,
                                                                n_train,
                                                                number_of_splits_testing_gt,
                                                                u_compactly_binarized_training,
                                                                u_compactly_binarized_testing,
                                                                u_training,
                                                                u_testing,
                                                                max_hamming_distance_tested,
                                                                sh_model.training_filename,
                                                                compress_type,
                                                                ordered_pcs,
                                                                False)
        log_file_test_destination += ".1"

        print("\nscore_precision")
        print(score_precision)
        print("score_recall")
        print(score_recall)
        print("score_f_measure")
        print(score_f_measure)
    else:
        # -- *** Evaluation, Approach 2: with a precise GT, instead of an approximation (done with reverse indices) -- #
        scores_reverse_indices = evaluate_with_reverse_indices(data_train_norm, data_test_norm, u_training, u_testing,
                                                               average_number_neighbors)
        log_file_test_destination += ".2"

        print("\nscores_reverse_indices")
        print(scores_reverse_indices)

    elapsed_time_formatted = time_process(start, time.time())

    rounding_dec = 10
    if eval_type == 0 or eval_type == 1:
        # -- Prepare results: For evaluations 1a and 1b -- #
        hdb_column = range(0, max_hamming_distance_tested)
        metrics_eval_1a_1b = np.round(np.column_stack([hdb_column, score_precision, score_recall, score_f_measure]), rounding_dec)
        # -- Log results -- #
        log_evaluation_results(log_file_test_destination, metrics_eval_1a_1b, max_hamming_distance_tested)

    else:
        # -- Prepare results: For evaluation 2 -- #
        metrics_eval_2 = np.round(np.column_stack([scores_reverse_indices]), rounding_dec)
        # -- Log results -- #
        log_evaluation_results(log_file_test_destination, metrics_eval_2, len(metrics_eval_2))


    # -- Check buckets' balance constraint -- #
    # check_buckets_balance_constraint(sh_model.n_bits, u_compactly_binarized_training, log_others)

    # -- Check how data falls into buckets && Calculate Hamming distance between all neighboring buckets -- #
    # all_unique_hashcodes = check_data_distribution_into_buckets(sh_model.n_bits, u_training, log_others)
    # calculate_hamming_dist_between_all_neighboring_buckets(all_unique_hashcodes)

    # -- Time and log testing phase -- #
    with open(log_others, 'a') as log_f:
        log_f.write("elapsed time for evaluation, for eval_type = " + log_file_test_destination + " => \n{0}\n\n".format(elapsed_time_formatted))
    # -- Close others log file -- #
    log_f.close()


main_test()

