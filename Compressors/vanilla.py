from helpers import *
import operator

# -- Vanilla SH Compression -- #
def compress(data_norm, sh_model, dataset_label):
    print("\n# -- VANILLA: {0} set -- #".format(dataset_label))

    # -- Get dataset dimensions -- #
    data_norm_n, data_norm_d = data_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_norm_pcaed = data_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_norm_pcaed_and_centered = data_norm_pcaed - np.tile(sh_model.mn, (data_norm_n, 1))
    # plot_2D(data_norm_pcaed_and_centered, False, "data_norm_pcaed_and_centered")

    omegas_compress_training = sh_model.modes * np.tile(sh_model.omega_zero, (sh_model.n_bits, 1))
    u = np.zeros((data_norm_n, sh_model.n_bits), dtype=bool)
    # test_hashcode = []
    # entire_data_box = np.zeros(())
    for ith_bit in range(0, sh_model.n_bits):
        omega_i = np.tile(omegas_compress_training[ith_bit, :], (data_norm_n, 1))

        data_box = data_norm_pcaed_and_centered * omega_i + math.pi / 2
        ys = np.sin(data_norm_pcaed_and_centered * omega_i + math.pi / 2)

        # -- START: Plot sine -- #
        #     plot_sine_partitioning_vanilla_sh(data_norm_pcaed_and_centered, omega_i, ys)
        # -- END: Plot sine -- #

        yi = np.prod(ys, axis=1)
        u[:, ith_bit] = yi < 0

    u_compactly_binarized = compact_bit_matrix(u)

    return u, u_compactly_binarized




