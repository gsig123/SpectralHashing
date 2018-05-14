from helpers import *
import operator

# Changes compared to original:
# Removed test_data_norm parameter. One seems to have been redundant.
def compress(data_train_norm, sh_model, dataset_label):
    print("\n# -- BALANCED, but with PC CUT REPEATED BIT EFFECT, where PCs are cut multiple times: {0} set -- #".format(dataset_label))
    # -- Define some params -- #
    corner_case_num_buckets = sh_model.n_bits  # * 128

    # -- Get dataset dimensions -- #
    data_train_norm_n, data_train_norm_d = data_train_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_train_norm_pcaed = data_train_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_train_norm_pcaed_and_centered = data_train_norm_pcaed - np.tile(sh_model.mn, (data_train_norm_n, 1))

    # -- This is for 'uord', but I think 'ord' will be excluded eventually -- #
    pcs_to_loop_through = enumerate(range(0, sh_model.n_bits))

    # -- Get data boxes -- #
    data_box_train = get_sine_data_box(data_train_norm_pcaed_and_centered, sh_model, data_train_norm_n)

    # -- Find out how many bits each pc contributes with -- #
    pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times = get_pcs_ith_bits_mapping(sh_model)
    pcs_we_store_bits_for = [tup[1] for tth, tup in enumerate(pcs_ith_bits_when_multiple_cuts)]

    if dataset_label == "testing":
        print_help("pcs_we_store_bits_for", pcs_we_store_bits_for)
        print("\npcs_ith_bits_mapping: {0}".format(pcs_ith_bits_mapping))
        print("\nsh_model.modes.shape: {0}".format(sh_model.modes.shape))
        print("\npcs_to_loop_through: {0}".format(range(0, sh_model.n_bits)))
        print("\npcs_ith_bits_when_multiple_cuts: {0}".format(pcs_ith_bits_when_multiple_cuts))
        print("\nfirst_pcs_when_axis_cut_multiple_times: {0}".format(first_pcs_when_axis_cut_multiple_times))

    data_hashcodes = [[] for _ in range(0, data_train_norm_n)]

    grey_codes_per_pc = {}
    for pth, pc in pcs_to_loop_through:
        if str(pc) in pcs_ith_bits_mapping.keys(): # quick fix. must be investigated
            num_bits_of_contribution = len(pcs_ith_bits_mapping[str(pc)]) if len(pcs_ith_bits_mapping[str(pc)]) > 0 else 1 # this is not the idea until the end and can confuse!
            num_buckets_per_pc = corner_case_num_buckets if np.power(2, num_bits_of_contribution) == 0 else np.power(2, num_bits_of_contribution)

            # -- Establish box information based only on training -- #
            pc_scores_train, max_pc_score_train, min_pc_score_train, range_pc_train, interval_pc_train = get_data_box_info(data_box_train, pc, num_buckets_per_pc)

            # -- GreyCode stuff -- #
            gray_codes_pc = list(GrayCode(num_bits_of_contribution).generate_gray())
            grey_codes_per_pc[str(pc)] = gray_codes_pc
            num_gray_codes = len(gray_codes_pc)

            # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
            pc_scores = data_box_train[:, pc]

            # -- Go through each data point in my pc box and check only the score corresponding to that pc dimension -- #
            for dp, pc_score in enumerate(pc_scores):
                if len(pcs_ith_bits_mapping[str(pc)]) > 0:
                    bucket_index = int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))
                    bucket_index = 0 if bucket_index < 0 else bucket_index
                    provenance_bucket_index = bucket_index if bucket_index < num_gray_codes else num_gray_codes - 1
                    bits_to_attach = [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]

                    data_hashcodes[dp] = np.hstack((data_hashcodes[dp], bits_to_attach))

    u = np.array(data_hashcodes, dtype=bool)
    u_compactly_binarized = compact_bit_matrix(u)

    return u, u_compactly_binarized
