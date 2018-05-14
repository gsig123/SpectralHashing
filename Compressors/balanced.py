from helpers import *
import operator


def decide_num_bits_from_each_participant_pc(pcs_ith_bits_mapping):
    num_bits_from_pcs = []
    for k, v in pcs_ith_bits_mapping.items():
        if len(v) > 0:
            num_bits_from_pcs.append((k, v[0]))

    print_help("num_bits_from_pcs", num_bits_from_pcs)
    return num_bits_from_pcs

# Changes compared to original:
# Removed test_data_norm parameter. One seems to have been redundant.
def compress(data_train_norm, sh_model, dataset_label):
    print("\n# -- BALANCED, AS INITIALLY INTENDED: {0} set -- #".format(dataset_label))
    # -- Decide num_bits we extract from each pc which has any modes in the first place (for pcs which in vanilla take their bits from several modes, we will directly select all bits at once, from the first ith mode in the values of the pc_key) -- #
    pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times = get_pcs_ith_bits_mapping(sh_model)
    print_help("pcs_ith_bits_mapping", pcs_ith_bits_mapping)
    print_help("pcs_ith_bits_when_multiple_cuts", pcs_ith_bits_when_multiple_cuts)
    print_help("first_pcs_when_axis_cut_multiple_times", first_pcs_when_axis_cut_multiple_times)
    pcs_vs_ith_mode = decide_num_bits_from_each_participant_pc(pcs_ith_bits_mapping)

    # -- Define some params -- #
    corner_case_num_buckets = sh_model.n_bits  # * 128

    # -- Get dataset dimensions -- #
    data_train_norm_n, data_train_norm_d = data_train_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_train_norm_pcaed = data_train_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_train_norm_pcaed_and_centered = data_train_norm_pcaed - np.tile(sh_model.mn, (data_train_norm_n, 1))

    # -- Loop through pcs as vanilla does, in the modes kind of order -- #
    pcs_to_loop_through = enumerate(range(0, sh_model.n_bits))

    # -- Create data box -- #
    data_box_train = get_sine_data_box(data_train_norm_pcaed_and_centered, sh_model, data_train_norm_n)

    data_hashcodes = [[] for _ in range(0, data_train_norm_n)]

    grey_codes_per_pc = {}
    for pth, pc in pcs_to_loop_through:
        if str(pc) in pcs_ith_bits_mapping.keys():
            # -- Establish n_buckets and n_bits -- #
            num_bits_of_contribution = len(pcs_ith_bits_mapping[str(pc)])
            if num_bits_of_contribution > 0:
                ith_mode_dim_to_extract_bits_from_for_pc = [pc_and_extraction_dim[1] for pc_and_extraction_dim in pcs_vs_ith_mode if pc_and_extraction_dim[0] == str(pc)][0]

                # Obs(*): In case num_buckets_per_pc, it means we have way too many buckets and their int value overflows
                num_buckets_per_pc = corner_case_num_buckets if np.power(2, num_bits_of_contribution) == 0 else np.power(2, num_bits_of_contribution)

                # -- Establish box information based only on training -- #
                pc_scores_train, max_pc_score_train, min_pc_score_train, range_pc_train, interval_pc_train = get_data_box_info(data_box_train, ith_mode_dim_to_extract_bits_from_for_pc, num_buckets_per_pc)

                # -- GreyCode stuff -- #
                gray_codes_pc = list(GrayCode(num_bits_of_contribution).generate_gray())
                grey_codes_per_pc[str(ith_mode_dim_to_extract_bits_from_for_pc)] = gray_codes_pc
                num_gray_codes = len(gray_codes_pc)

                # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
                pc_scores = data_box_train[:, ith_mode_dim_to_extract_bits_from_for_pc]

                # -- Go through each data point in my pc box and check only the score corresponding to that pc dimension -- #
                for dp, pc_score in enumerate(pc_scores):
                    provenance_bucket_index = get_provenance_bucket_index(pc_score, min_pc_score_train, interval_pc_train, num_gray_codes)
                    bits = [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]
                    data_hashcodes[dp] = np.hstack((data_hashcodes[dp], bits))


    u = np.array(data_hashcodes, dtype=bool)
    u_compactly_binarized = compact_bit_matrix(u)

    return u, u_compactly_binarized
