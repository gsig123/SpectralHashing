class SHParam(object):
    def __init__(self, mn, mx, modes, n_bits):
        self.mn = mn
        self.mx = mx
        self.modes = modes
        self.n_bits = n_bits

    def __set_mn__(self, mn):
        self.mn = mn

    def __set_mx__(self, mx):
        self.mx = mx

    def __set_modes__(self, modes):
        self.modes = modes

    def __set_n_bits__(self, n_bits):
        self.n_bits = n_bits

    def __str__(self):
        return str(self.mn) + ", " + str(self.mx) + ", " + str(self.modes) + ", " + str(self.n_bits)



class SHModel(object):
    def __init__(self, mn, mx, modes, n_bits, pc_from_training, omega_zero, training_filename):
        self.mn = mn
        self.mx = mx
        self.modes = modes
        self.n_bits = n_bits
        self.pc_from_training = pc_from_training
        self.omega_zero = omega_zero
        self.training_filename = training_filename

    def __set_mn__(self, mn):
        self.mn = mn

    def __set_mx__(self, mx):
        self.mx = mx

    def __set_modes__(self, modes):
        self.modes = modes

    def __set_n_bits__(self, n_bits):
        self.n_bits = n_bits

    def __set_pc_from_training__(self, pc_from_training):
        self.pc_from_training = pc_from_training

    def __set_omega_zero__(self, omega_zero):
        self.omega_zero = omega_zero

    def __set_training_filename__(self, training_filename):
        self.training_filename = training_filename

    def __str__(self):
        return str(self.mn) + ", " + str(self.mx) + ", " + str(self.modes) + ", " + str(self.n_bits) + ", " + str(self.training_filename) + ", " + str(self.pc_from_training)



class ApproxGT(object):
    def __init__(self, w_true_test_training, d_ball_eucl, average_number_neighbors, n_train, n_test, approx_gt_filename):
        self.w_true_test_training = w_true_test_training
        self.d_ball_eucl = d_ball_eucl
        self.average_number_neighbors = average_number_neighbors
        self.n_train = n_train
        self.n_test = n_test
        self.approx_gt_filename = approx_gt_filename

    def __set_w_true_test_training__(self, w_true_test_training):
        self.w_true_test_training = w_true_test_training

    def __set_d_ball_eucl__(self, d_ball_eucl):
        self.d_ball = d_ball_eucl

    def __set_average_number_neighbors__(self, average_number_neighbors):
        self.average_number_neighbors = average_number_neighbors

    def __set_n_train__(self, n_train):
        self.n_train = n_train

    def __set_n_test__(self, n_test):
        self.n_test = n_test

    def __set_approx_gt_filename__(self, approx_gt_filename):
        self.approx_gt_filename = approx_gt_filename

    def __str__(self):
        return str(self.d_ball_eucl) + ", " + str(self.n_train) + ", " + str(self.n_test) + ", " + str(self.average_number_neighbors)+ ", " + str(self.indices_train_test_files_concatenated)

