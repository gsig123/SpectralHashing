from helpers import *
import operator

def decide_pcs_bit_dominance(sh_model):
    # -- Find pcs dominance order, where modes order decides that, really (based on the assumption of the eigenvalues) -- #
    pcs_ordered_by_dominance = []
    for ith_mode, mode in enumerate(sh_model.modes):
        dominant_pc_per_mode = [m for m, mode_val in enumerate(mode) if mode_val != 0]
        pcs_ordered_by_dominance.append(dominant_pc_per_mode[0])

    print_help("pcs_ordered_by_dominance", pcs_ordered_by_dominance)
    # print(len(pcs_ordered_by_dominance))

    # -- Assign decreasing num_bits to pcs, according to their established dominance -- #
    num_bits_pcs = []
    bits_per_pcs = get_pc_bitwise_contribution(sh_model)
    for v in bits_per_pcs.values():
        num_bits_pcs.append(len(v))
    num_bits_pcs.sort(reverse=True)

    # print_help("num_bits_pcs", num_bits_pcs)
    limit = len(num_bits_pcs)

    pcs_ordered_by_dominance_limited = pcs_ordered_by_dominance[:limit]

    num_bits_from_pcs = []
    for p, (pc, num_bits) in enumerate(zip(pcs_ordered_by_dominance_limited, num_bits_pcs)):
        surplus = 1
        # surplus = p / (p * 1.0 + 1.0) + 1
        # surplus = surplus if num_bits > 2 else 1
        num_bits_from_pcs.append((pc, int(num_bits + surplus)))

    print_help("num_bits_from_pcs", num_bits_from_pcs)

    return num_bits_from_pcs, pcs_ordered_by_dominance_limited




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
    print("\n# -- PCDOMMODESORDER ACTUAL PCS (compress_dataset__pc_dominance_by_modes_order), but where the ith pc is assigned (ith_num_bits + 1), such that ith_num_bits corresponds to the ith num_bits of the ith pc according to modes' order (if by modes order the ith pc is the ith most important, then we'll look at the ith num_bits we'd normally extract and + 1, because it seems this weird parametrization might increase the scores): {0} set -- #".format(dataset_label))
    pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times = get_pcs_ith_bits_mapping(
        sh_model)
    pcs_vs_ith_mode = decide_num_bits_from_each_participant_pc(pcs_ith_bits_mapping)

    # -- Decide num_bits of contribution for each pc (depending on modes order) -- #
    num_bits_from_pcs, pcs_ordered_by_dominance_limited = decide_pcs_bit_dominance(sh_model)

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
    bits_per_pcs = get_pc_bitwise_contribution(sh_model)
    # print("\nbits_per_pcs: {0}".format(bits_per_pcs))

    pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times = get_pcs_ith_bits_mapping(sh_model)
    pcs_we_store_bits_for = [tup[1] for tth, tup in enumerate(pcs_ith_bits_when_multiple_cuts)]

    if dataset_label == "testing":
        print_help("pcs_we_store_bits_for", pcs_we_store_bits_for)
        print("\npcs_ith_bits_mapping: {0}".format(pcs_ith_bits_mapping))
        print("\npcs_to_loop_through: {0}".format(range(0, sh_model.n_bits)))
        print("\npcs_ith_bits_when_multiple_cuts: {0}".format(pcs_ith_bits_when_multiple_cuts))
        print("\nfirst_pcs_when_axis_cut_multiple_times: {0}".format(first_pcs_when_axis_cut_multiple_times))

    data_hashcodes = [[] for _ in range(0, data_train_norm_n)]

    grey_codes_per_pc = {}
    for pth, pc in pcs_to_loop_through:
        if pc in pcs_ordered_by_dominance_limited:
            # num_bits_of_contribution = len(pcs_ith_bits_mapping[str(pc)]) if len(pcs_ith_bits_mapping[str(pc)]) > 0 else 1
            num_bits_of_contribution = [pc_mode_pair[1] for pc_mode_pair in num_bits_from_pcs if pc == pc_mode_pair[0]][0]

            num_buckets_per_pc = corner_case_num_buckets if np.power(2, num_bits_of_contribution) == 0 else np.power(2, num_bits_of_contribution)

            ith_mode_dim_to_extract_bits_from_for_pc = \
            [pc_and_extraction_dim[1] for pc_and_extraction_dim in pcs_vs_ith_mode if
             pc_and_extraction_dim[0] == str(pc)][0]

            print("PC, NUM_BITS, MODE_FROM: {0}, {1}, {2}".format(pc, num_bits_of_contribution, ith_mode_dim_to_extract_bits_from_for_pc))

            # -- Establish box information based only on training -- #
            pc_scores_train, max_pc_score_train, min_pc_score_train, range_pc_train, interval_pc_train = get_data_box_info(
                data_box_train, ith_mode_dim_to_extract_bits_from_for_pc, num_buckets_per_pc)

            # -- GreyCode stuff -- #
            gray_codes_pc = list(GrayCode(num_bits_of_contribution).generate_gray())
            grey_codes_per_pc[str(pc)] = gray_codes_pc
            num_gray_codes = len(gray_codes_pc)

            # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
            pc_scores = data_box_train[:, ith_mode_dim_to_extract_bits_from_for_pc]
            # -- Go through each data point in my pc box and check only the score corresponding to that pc dimension -- #
            for dp, pc_score in enumerate(pc_scores):
                if str(pc) in pcs_ith_bits_mapping.keys(): # quick fix. must be investigated
                    if len(pcs_ith_bits_mapping[str(pc)]) > 0:
                        bucket_index = int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))
                        bucket_index = 0 if bucket_index < 0 else bucket_index
                        provenance_bucket_index = bucket_index if bucket_index < num_gray_codes else num_gray_codes - 1
                        bits_to_attach = [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]

                        # -- Limit hashcode length to whatever needed initially, even if it exceeds -- #
                        if len(data_hashcodes[dp]) < sh_model.n_bits:
                            data_hashcodes[dp] = np.hstack((data_hashcodes[dp], bits_to_attach))
                        else:
                            continue

    u = np.array(data_hashcodes, dtype=bool)
    u_compactly_binarized = compact_bit_matrix(u)

    return u, u_compactly_binarized





