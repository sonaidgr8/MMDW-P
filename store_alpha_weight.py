import numpy as np


class store_alpha_weight:

    def __init__(self, config, dataset, train_ids):
        self.nr_class = dataset.n_labels
        self.dimension_svm = int(config.L_COMPONENTS)
        self.weight_b = np.zeros((self.nr_class, int(config.L_COMPONENTS)))
        self.alpha_b = np.zeros((np.count_nonzero(train_ids), self.nr_class))
