class EvalDebugApproachComponents(object):
    def __init__(self):
        print("Initialized empty EvalDebugApproachComponents object.")

    def __set_hamm_dist_debug__(self, hamm_dist_debug):
        self.hamm_dist_debug = hamm_dist_debug

    def __set_indices_totals__(self, retrieved_good_pairs, retrieved_pairs, total_good_pairs):
        self.retrieved_good_pairs = retrieved_good_pairs
        self.total_good_pairs = total_good_pairs
        self.retrieved_pairs = retrieved_pairs

    def __set_compress_type__(self, compress_type):
        self.compress_type = compress_type

    def __set_unique_buckets_and_indices__(self, unique_buckets_and_indices_training, unique_buckets_and_indices_testing):
        self.unique_buckets_and_indices_training = unique_buckets_and_indices_training
        self.unique_buckets_and_indices_testing = unique_buckets_and_indices_testing

    def __set_u_training_and_u_testing__(self, u_training, u_testing):
        self.u_training = u_training
        self.u_testing = u_testing


    def __set_indices__(self, gt_nn_indices, indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl, indices_pairs_of_good_pairs_in_d_hamm_for_all_queries):
        self.indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl = indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl
        self.indices_pairs_of_good_pairs_in_d_hamm_for_all_queries = indices_pairs_of_good_pairs_in_d_hamm_for_all_queries
        self.gt_nn_indices = gt_nn_indices

    def __str__(self):
        return str(self.hamm_dist_debug) + "; \n" + str(self.compress_type)

