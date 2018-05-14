import numpy as np
import pickle

# python debug.py -input "./Results/Tests__Why_is_balanced_worse_than_vanilla/Logs/randomblob_5" -compress_types="vanilla_balanced"

def read_args():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-input", help="input training set", required=1)
    parser.add_argument("-compress_types", help="methods to compare; e.g.: vanilla_balanced", type=str, required=1)
    args = parser.parse_args()

    return args

def print_buckets(buckets):
    for k, val in buckets.items():
        print("bucket: " + k + ", with {0} points' indices: {1}".format(len(val), val))


def print_associated_buckets_for_symm_diff_indices(buckets_dict, symm_diff_indices):
    for diff_index in symm_diff_indices:
        for k, v in buckets_dict.items():
            if diff_index in v:
                print("nn index {0} has actual hashcode {1}".format(diff_index, k))


        # print([k for k, v in buckets_dict.items() if v == diff_index])
        # print("nn index {0} has actual hashcode {1}".format(diff_index, buckets_dict.keys()[buckets_dict.values().index(diff_index)]))

    if len(symm_diff_indices) > 0:
        print("\n")



def debug_bucket_indices_symmetric_difference(b1_buckets, b2_buckets, bucket_1_name, bucket_2_name, label):
    for b1_b, b2_b in zip(b1_buckets, b2_buckets):
        print("\nFor " + bucket_1_name + "/" + bucket_2_name + " bucket {0} / {1} =>".format(b1_b, b2_b))
        # print(bucket_1_name + "_bucket indices: {0}\n".format(b1_buckets[b1_b]) + bucket_2_name + "_bucket indices: {0}".format(b2_buckets[b2_b]))
        b1_bucket = b1_buckets[b1_b]
        b2_bucket = b2_buckets[b2_b]
        b1_minus_b2 = list(set(b1_bucket) - set(b2_bucket))
        b2_minus_b1 = list(set(b2_bucket) - set(b1_bucket))
        if len(b1_minus_b2) > 0:
            print("nn indices which " + bucket_1_name + " bucket has, but " + bucket_2_name + " bucket doesn't => {0}".format(b1_minus_b2))
            print_associated_buckets_for_symm_diff_indices(b1_buckets, b1_minus_b2)
        if len(b2_minus_b1) > 0:
            print("nn indices which " + bucket_1_name + " bucket has, but " + bucket_2_name + " bucket doesn't => {0}".format(b2_minus_b1))
            print_associated_buckets_for_symm_diff_indices(b2_buckets, b2_minus_b1)

        if len(b1_minus_b2) == 0 and len(b1_minus_b2) == len(b2_minus_b1):
            print("Obs(***) For {0}, all indices got hashed to the exact same bucket!".format(label))


def print_gt_nn_indices(gt_nn_indices):
    for ith, gt_nn_indices_ith in enumerate(gt_nn_indices):
        print("{0}th, #{1}: {2}".format(ith, len(gt_nn_indices_ith), gt_nn_indices_ith))


def debug_gt_nn_indices(c1_gt_nn_indices, c2_gt_nn_indices, compress_type_1, compress_type_2):
    all_identical = True
    for c1_row, c2_row in zip(c1_gt_nn_indices, c2_gt_nn_indices):
        c1_minus_c2 = list(set(c1_row) - set(c2_row))
        c2_minus_c1 = list(set(c2_row) - set(c1_row))

        if len(c1_minus_c2) > 0:
            print("\ngt_nn_indices which {0} has and {1} doesn't are: {2}".format(compress_type_1, compress_type_2, c1_minus_c2))
            all_identical = False
        if len(c2_minus_c1) > 0:
            print("\ngt_nn_indices which {0} has and {1} doesn't are: {2}".format(compress_type_2, compress_type_2, c2_minus_c1))
            all_identical = False

    if all_identical:
        print("\ngt_nn_indices are all identical for both methods!")


def debug_hamm_indices(
        c1_hamm_indices,
        c2_hamm_indices, compress_type_1, compress_type_2, r_pairs_c1, r_pairs_c2):
    counter_c1 = 0
    counter_c2 = 0
    query_points_for_which_indices_differ = []

    for ith, (c1_row, c2_row) in enumerate(zip(c1_hamm_indices, c2_hamm_indices)):
        c1_minus_c2 = list(set(c1_row) - set(c2_row))
        c2_minus_c1 = list(set(c2_row) - set(c1_row))

        counter_c1 += len(c1_row)
        counter_c2 += len(c2_row)

        if len(c1_minus_c2) > 0:
            print("\n*** retrieved_pairs (all pairs in Hamming in the hamm_ball_debug) which {0} has and {1} doesn't are: {2}".format(compress_type_1, compress_type_2, c1_minus_c2))
        if len(c2_minus_c1) > 0:
            print("*** retrieved_pairs (all pairs in Hamming in the hamm_ball_debug) which {0} has and {1} doesn't are: {2}".format(compress_type_2, compress_type_2, c2_minus_c1))
        if len(c1_minus_c2) > 0 and len(c2_minus_c1) > 0:
            query_points_for_which_indices_differ.append(ith)

    print("\nFinal counter check for retrieved_pairs({0})={1} and retrieved_pairs({2})={3} =>".format(counter_c1, compress_type_1, counter_c2, compress_type_2))
    print("\nQuery points for which retrieved_pairs differ are #{0} => {1}".format(len(query_points_for_which_indices_differ), query_points_for_which_indices_differ))
    print(r_pairs_c1, r_pairs_c2)


def debug_hashcodes(c1_u, c2_u, label, compress_type_1, compress_type_2):
    hashcodes_not_equal = []
    for ith, (c1_hashcode, c2_hashcode) in enumerate(zip(c1_u, c2_u)):
        if not c1_hashcode == c2_hashcode:
            hashcodes_not_equal.append(ith)

    if len(hashcodes_not_equal) > 0:
        print("\nHashcodes not equal for {0} sets are =>".format(label))
        for hc in hashcodes_not_equal:
            if hc == 47:
                print("{0}th query point: {1} ({2}) and {3} ({4})".format(hc, c1_u[hc], compress_type_1, c2_u[hc], compress_type_2))
                for ith, (c1_char, c2_char) in enumerate(zip(c1_u[hc], c2_u[hc])):
                    if c1_char != c2_char:
                        print("(*) ---- {0}th PC => v={1} => b={2}\n".format(ith+1, c1_char, c2_char))
    else:
        print("\nAll hashcodes seem to be equal for {0} set\n".format(label))


def debug_evaluation():
    # -- Print generalities -- #
    print("\nWe search in hamm_ball_debug => {0}".format(eval_debug_object_1.hamm_dist_debug))
    print("\nComparison => {0} vs. {1}".format(eval_debug_object_1.compress_type, eval_debug_object_2.compress_type))
    # -- 1. Debug Buckets' indices for b1 vs. b2 -- #
    b1_buckets_training = eval_debug_object_1.unique_buckets_and_indices_training
    b2_buckets_training = eval_debug_object_2.unique_buckets_and_indices_training
    debug_bucket_indices_symmetric_difference(b1_buckets_training, b2_buckets_training, compress_type_1, compress_type_2, "training")

    b1_buckets_testing = eval_debug_object_1.unique_buckets_and_indices_testing
    b2_buckets_testing = eval_debug_object_2.unique_buckets_and_indices_testing
    debug_bucket_indices_symmetric_difference(b1_buckets_testing, b2_buckets_testing, compress_type_1, compress_type_2, "testing")

    for b1_b, b2_b in zip(b1_buckets_testing, b2_buckets_testing):
        print(b1_b, b2_b)
    print(b1_buckets_testing)
    print(b2_buckets_testing)


    # -- 2. Debug gt_nn_indices for compress_type_1 vs. compress_type_2 -- #
    c1_gt_nn_indices = eval_debug_object_1.gt_nn_indices
    c2_gt_nn_indices = eval_debug_object_2.gt_nn_indices

    debug_gt_nn_indices(c1_gt_nn_indices, c2_gt_nn_indices, compress_type_1, compress_type_2)

    # -- 3. Debug indices_pairs_of_good_pairs_in_d_hamm_for_all_queries for compress_type_1 vs. compress_type_2 -- #
    c1_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries = eval_debug_object_1.indices_pairs_of_good_pairs_in_d_hamm_for_all_queries
    c2_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries = eval_debug_object_2.indices_pairs_of_good_pairs_in_d_hamm_for_all_queries
    r_pairs_c1 = eval_debug_object_1.retrieved_pairs
    r_pairs_c2 = eval_debug_object_2.retrieved_pairs

    debug_hamm_indices(c1_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries, c2_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries, compress_type_1, compress_type_2, r_pairs_c1, r_pairs_c2)

    # -- 3. Debug indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl for compress_type_1 vs. compress_type_2 -- #
    c1_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl = eval_debug_object_1.indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl
    c2_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl = eval_debug_object_2.indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl
    r_good_pairs_c1 = eval_debug_object_1.retrieved_good_pairs
    r_good_pairs_c2 = eval_debug_object_2.retrieved_good_pairs

    debug_hamm_indices(
        c1_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl,
        c2_indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl, compress_type_1, compress_type_2, r_good_pairs_c1,
        r_good_pairs_c2)

    # -- 4. Check hashcodes -- #
    c1_u_training = eval_debug_object_1.u_training
    c2_u_training = eval_debug_object_2.u_training
    c1_u_testing = eval_debug_object_1.u_testing
    c2_u_testing = eval_debug_object_2.u_testing

    debug_hashcodes(c1_u_training, c2_u_training, "training", compress_type_1, compress_type_2)
    debug_hashcodes(c1_u_testing, c2_u_testing, "testing", compress_type_1, compress_type_2)

    # -- 5. Others -- #
    print("\n total_good_pairs_1={0}, total_good_pairs_2={1}".format(eval_debug_object_1.total_good_pairs, eval_debug_object_2.total_good_pairs))



if __name__ == '__main__':
    # -- Read args -- #
    args = read_args()
    input_filename = args.input
    compress_types = args.compress_types

    types = compress_types.split("_")
    compress_type_1 = types[0]
    compress_type_2 = types[1]

    compress_type_1_filename = input_filename + ".train.eval.debug." + compress_type_1
    compress_type_2_filename = input_filename + ".train.eval.debug." + compress_type_2


    eval_debug_object_1 = pickle.load(open(compress_type_1_filename, "rb"))
    eval_debug_object_2 = pickle.load(open(compress_type_2_filename, "rb"))

    # -- DEBUG: Evaluation issues (e.g.: different hashcodes etc.) -- #
    debug_evaluation()

    # -- DEBUG: data_box issues -- #
    compress_type_1_filename_db = input_filename + ".train.databox.debug." + compress_type_1
    compress_type_2_filename_db = input_filename + ".train.databox.debug." + compress_type_2

