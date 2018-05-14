from helpers import *
import operator


def decide_num_bits_contribution_per_pc(modes):
    num_bits_from_pcs = []
    for ith_pc, pc_mode_cut in enumerate(modes.T):
        pc_mode_sequence = [(pc, mode) for pc, mode in enumerate(pc_mode_cut) if mode != 0]
        max_mode = len(pc_mode_sequence)
        cuts_range = list(reversed(range(1, max_mode + 1)))
        pc_mode_sequence = [(pc_mode_pair[0], cuts_range[ith]) for ith, pc_mode_pair in enumerate(pc_mode_sequence)]
        if len(pc_mode_sequence) > 0:
            num_bits_from_pcs.extend(pc_mode_sequence)
            print("(pc, mode): {0}".format(pc_mode_sequence))
            # print("\n\n")

    all_num_bits = 0
    for pair in num_bits_from_pcs:
        all_num_bits += pair[1]
    print_help("num_bits_from_pcs", num_bits_from_pcs)
    print_help("all_num_bits", all_num_bits)

    return num_bits_from_pcs

# Changes compared to original:
# Removed test_data_norm parameter. One seems to have been redundant.
def compress(data_train_norm, sh_model, dataset_label):
    print("\n# -- BALANCED (compress_dataset__pc_cut_repeated_multiple_bits), but with PC CUT REPEATED BIT EFFECT AAAAND trying to get a num of bis identical to the mode for each pc, when PCs are cut multiple times: {0} set -- #".format(dataset_label))
    # -- Decide num_bits of contribution for each pc which is cut multiple times -- #
    num_bits_from_pcs = decide_num_bits_contribution_per_pc(sh_model.modes)

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

    total_bits = 0
    grey_codes_per_pc = {}
    for pth, pc in pcs_to_loop_through:
        # num_bits_of_contribution = len(pcs_ith_bits_mapping[str(pc)]) if len(pcs_ith_bits_mapping[str(pc)]) > 0 else 1
        num_bits_of_contribution = [pc_mode_pair[1] for pc_mode_pair in num_bits_from_pcs if pc == pc_mode_pair[0]][0]
        # print("PC, NUM_BITS: {0}, {1}".format(pc, num_bits_of_contribution))

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



