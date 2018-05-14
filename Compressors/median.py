from helpers import *

# -- Median SH Compression -- #
def decide_num_bits_from_each_participant_pc(pcs_ith_bits_mapping):
    num_bits_from_pcs = []
    for k, v in pcs_ith_bits_mapping.items():
        if len(v) > 0:
            num_bits_from_pcs.append((k, v[0]))

    print_help("num_bits_from_pcs", num_bits_from_pcs)
    return num_bits_from_pcs

# Changes compared to original:
# Removed test_data_norm parameter. One seems to have been redundant.
def compress(data_norm, sh_model, dataset_label):
    print("\n# -- MEDIAN CORRECTED, I hope: {0} set -- #".format(
        dataset_label))
    pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times = get_pcs_ith_bits_mapping(sh_model)
    print_help("pcs_ith_bits_mapping", pcs_ith_bits_mapping)
    print_help("pcs_ith_bits_when_multiple_cuts", pcs_ith_bits_when_multiple_cuts)
    print_help("first_pcs_when_axis_cut_multiple_times", first_pcs_when_axis_cut_multiple_times)
    pcs_vs_ith_mode = decide_num_bits_from_each_participant_pc(pcs_ith_bits_mapping)

    # -- Define some params -- #
    corner_case_num_buckets = sh_model.n_bits  # * 128

    # -- Get dataset dimensions -- #
    data_norm_n, data_train_norm_d = data_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_norm_pcaed = data_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_norm_pcaed_and_centered = data_norm_pcaed - np.tile(sh_model.mn, (data_norm_n, 1))

    # -- Loop through pcs as vanilla does, in the modes kind of order -- #
    pcs_to_loop_through = enumerate(range(0, sh_model.n_bits))

    # -- Create data box -- #
    data_box_train = get_sine_data_box(data_norm_pcaed_and_centered, sh_model, data_norm_n)

    data_hashcodes = [[] for _ in range(0, data_norm_n)]

    grey_codes_per_pc = {}
    for pth, pc in pcs_to_loop_through:
        if str(pc) in pcs_ith_bits_mapping.keys():
            # -- Establish n_buckets and n_bits -- #
            num_bits_of_contribution = len(pcs_ith_bits_mapping[str(pc)])
            if num_bits_of_contribution > 0:
                ith_mode_dim_to_extract_bits_from_for_pc = [pc_and_extraction_dim[1] for pc_and_extraction_dim in pcs_vs_ith_mode if pc_and_extraction_dim[0] == str(pc)][0]

                # Obs(*): In case num_buckets_per_pc, it means we have way too many buckets and their int value overflows
                num_buckets_per_pc = corner_case_num_buckets if np.power(2, num_bits_of_contribution) == 0 else np.power(2, num_bits_of_contribution)

                # -- GreyCode stuff -- #
                gray_codes_pc = list(GrayCode(num_bits_of_contribution).generate_gray())
                grey_codes_per_pc[str(ith_mode_dim_to_extract_bits_from_for_pc)] = gray_codes_pc
                num_gray_codes = len(gray_codes_pc)

                # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
                pc_scores = data_box_train[:, ith_mode_dim_to_extract_bits_from_for_pc]

                # -- Plot box/PC partitions -- #
                median_cut_points = find_median_recursively__short(pc_scores, num_buckets_per_pc - 1)

                for dp, pc_score in enumerate(pc_scores):
                    closest_to_median_cut_point = min(median_cut_points, key=lambda x: abs(x - pc_score))
                    bucket_index = median_cut_points.index(closest_to_median_cut_point)
                    if bucket_index < 0:
                        bucket_index = 0
                    provenance_bucket_index = bucket_index if pc_score < closest_to_median_cut_point else bucket_index + 1
                    bits_to_attach = [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]
                    data_hashcodes[dp] = np.hstack((data_hashcodes[dp], bits_to_attach))

            u = np.array(data_hashcodes, dtype=bool)
            u_compactly_binarized = compact_bit_matrix(u)

    return u, u_compactly_binarized
