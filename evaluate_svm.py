import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from store_alpha_weight import store_alpha_weight
from main_algo import construct_indicator, get_perf_metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class evaluate_svm:
    data_num = 0
    dimension = 0
    C = 10.0
    eps = 0.1
    limit_random = 0.0
    test_cnt = 0
    train_cnt = 0
    val_cnt = 0

    def __init__(self, config, dataset, logger, W, train_ids, val_ids, test_ids):
        self.data_num = dataset.n_ids
        self.dimension = int(config.L_COMPONENTS)
        self.Wt = W
        self.train_cnt = np.count_nonzero(train_ids)
        self.val_cnt = np.count_nonzero(val_ids)
        self.test_cnt = np.count_nonzero(test_ids)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

    def train_evaluate_svm(self, config, dataset, logger):
        saw = store_alpha_weight(config, dataset, self.train_ids)

        clf = OneVsRestClassifier(LinearSVC(random_state=0, tol=self.eps, C=self.C))
        entity_embedding = self.Wt
        clf.fit(entity_embedding[self.train_ids, :], dataset.truth[self.train_ids, :])
        predictions = clf.decision_function(entity_embedding[self.test_ids, :])
        y_pred = construct_indicator(predictions, dataset.truth[self.test_ids, :])
        mi = f1_score(dataset.truth[self.test_ids, :], y_pred, average="micro")
        ma = f1_score(dataset.truth[self.test_ids, :], y_pred, average="macro")
        acc = accuracy_score(dataset.truth[self.test_ids, :], y_pred)

        clf_norm = OneVsRestClassifier(LinearSVC(random_state=0, tol=self.eps, C=self.C))
        entity_embedding_norm = normalize(entity_embedding, axis=1, norm='l2')
        clf_norm.fit(entity_embedding_norm[self.train_ids, :], dataset.truth[self.train_ids, :])
        predictions_norm = clf_norm.decision_function(entity_embedding_norm[self.test_ids, :])
        y_pred_norm = construct_indicator(predictions_norm, dataset.truth[self.test_ids, :])
        mi_norm = f1_score(dataset.truth[self.test_ids, :], y_pred_norm, average="micro")
        ma_norm = f1_score(dataset.truth[self.test_ids, :], y_pred_norm, average="macro")
        acc_norm = accuracy_score(dataset.truth[self.test_ids, :], y_pred_norm)

        if mi >= mi_norm:
            performances = {'accuracy': acc, 'micro_f1': mi, 'macro_f1': ma}
            saw.weight_b = clf.coef_
            saw.alpha_b[:, 0: saw.nr_class] = clf.intercept_.flatten()
        else:
            performances_norm = {'accuracy': acc_norm, 'micro_f1': mi_norm, 'macro_f1': ma_norm}
            saw.weight_b = clf_norm.coef_
            saw.alpha_b[:, 0: saw.nr_class] = clf_norm.intercept_.flatten()
            performances = performances_norm

        print("*************Precision = ", performances['micro_f1'] * 100, "%*************")
        return saw, performances

