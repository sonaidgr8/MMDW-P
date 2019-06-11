import numpy as np
import time
from store_alpha_weight import store_alpha_weight
from evaluate_svm import evaluate_svm
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler, normalize
from os import path
RESULT_DICT = {'Q': None, 'U': None, 'i': 0}

class learn_w:
    lmbda = 0.0
    flagnum = 0
    alpha = 0.0
    alpha_bias = 0.0
    cost = 0.0
    limit_random = 0.0
    alpha_level = 0

    init = True
    write_vec = True
    test_switch = False
    steps_init = 20
    steps_after = 0

    def __init__(self, config, dataset, train_ids):
        self.graph = np.zeros((dataset.n_ids, dataset.n_ids), dtype=np.float32)
        self.E = np.zeros((dataset.n_ids, int(config.L_COMPONENTS)), dtype=np.float32)
        self.F = np.zeros((int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
        self.bias = np.zeros((dataset.n_ids, int(config.L_COMPONENTS)), dtype=np.float32)
        self.saw = store_alpha_weight(config, dataset, train_ids)
        self.train_ids = train_ids

    def make_graph(self, config, dataset, logger, S):
        self.graph = S
        np.random.seed(0)
        self.E = np.array(np.random.rand(dataset.n_ids, int(config.L_COMPONENTS)), dtype=np.float32)
        self.F = np.array(np.random.rand(int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
        self.E = normalize(self.E, axis=1, norm='l2')
        self.F = normalize(self.F, axis=1, norm='l2')
        self.train_w(config, dataset, logger)
        self.init = False

    def pre_compute_bias(self, config, dataset):
        alpha_b = self.saw.alpha_b
        weight_b = self.saw.weight_b
        train_indices = np.where(self.train_ids == True)[0].tolist()
        label_dict = dict.fromkeys(set(train_indices), [])
        for k in label_dict.keys():
            label_dict[k] = np.where(dataset.truth[k] == 1)[0].tolist()
        mask = dataset.truth[self.train_ids] == 1
        c = np.zeros(self.saw.alpha_b.shape)
        c[mask] += evaluate_svm.C
        i = 0
        for id in train_indices:
            for label in label_dict[id]:
                for j in range(self.saw.nr_class):
                    self.bias[id][0: self.saw.dimension_svm] = (c[i][j] - alpha_b[i][j]) * (
                         weight_b[label][0: self.saw.dimension_svm] - weight_b[j][0: self.saw.dimension_svm])
            if norm(self.bias[id]) != 0:
                self.bias[id] = self.bias[id] / norm(self.bias[id])
            i += 1

    def train_w(self, config, dataset, logger):
        last_loss = 0.0
        alpha = float(config.ALPHA)
        if (not self.init):
            self.pre_compute_bias(config, dataset)
        if (self.init):
            for step in range(self.steps_init):
                # print("iter = ", step,'\t', "Process 1 for training",'\t', (step * 100) / self.steps_init, "%")
                B = np.copy(self.F)
                drv = np.zeros((int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
                Hess = np.zeros((int(config.L_COMPONENTS), int(config.L_COMPONENTS)), dtype=np.float32)
                drv = drv - 2 * alpha * np.dot(B, self.graph.T) + 2 * alpha * np.dot(np.dot(B, B.T), self.E.T) + 2 * self.lmbda * self.E.T
                Hess = Hess + 2 * alpha * np.dot(B, B.T)
                np.fill_diagonal(Hess, np.diagonal(Hess) + 2 * self.lmbda)
                rt = ((- drv.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                dt = rt
                drv.fill(0.0)
                vecE = (self.E.flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                # Update E
                countE = 0
                while (True):
                    countE += 1
                    norm_rt = norm(rt)
                    if countE > 10:
                        break
                    else:
                        storeHessdtS = np.zeros((int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
                        dtS = dt.reshape((dataset.n_ids, int(config.L_COMPONENTS))).T
                        storeHessdtS = storeHessdtS + np.dot(Hess, dtS)
                        Hdt = ((storeHessdtS.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                        rtrt = norm(rt) ** 2
                        dtHdt = np.sum(np.multiply(dt, Hdt))
                        at = rtrt / dtHdt
                        vecE = vecE + at * dt
                        rt = rt - at * Hdt
                        rtmprtmp = rtrt
                        rtrt += norm(rt) ** 2
                        bt = rtrt / rtmprtmp
                        dt = rt + bt * dt
                self.E = vecE.reshape((dataset.n_ids, int(config.L_COMPONENTS)))

                # Update F
                drv = drv + 2 * alpha * np.dot(np.dot(self.E.T, self.E), self.F) - 2 * alpha * np.dot(self.E.T, self.graph) + 2 * self.lmbda * self.F
                rt = ((- drv.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                dt = rt
                drv.fill(0.0)
                vecF = (self.F.T).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                countF = 0
                while (True):
                    countF += 1
                    norm_rt = norm(rt)
                    if countF > 10:
                        break
                    else:
                        dtS = dt.reshape((dataset.n_ids, int(config.L_COMPONENTS))).T
                        storeHdt = 2 * alpha * np.dot(np.dot(self.E.T, self.E), dtS) + 2 * self.lmbda * dtS
                        Hdt = ((storeHdt.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                        rtrt = norm(rt) ** 2
                        dtHdt = np.sum(np.multiply(dt, Hdt))
                        at = rtrt / dtHdt
                        vecF = vecF + at * dt
                        rt = rt - at * Hdt
                        rtmprtmp = rtrt
                        rtrt += norm(rt) ** 2
                        bt = rtrt / rtmprtmp
                        dt = rt + bt * dt
                Ft = (vecF.flatten()).reshape((dataset.n_ids, int(config.L_COMPONENTS)))
                self.F = Ft.T

                EF = np.dot(self.E, self.F)
                fitMEF = norm(self.graph - EF) ** 2
                loss = alpha * fitMEF + (self.lmbda) * (norm(self.E) ** 2 + norm(self.F) ** 2)
                if (abs(last_loss - loss) < float(config.CONV_LS)):
                    break
                last_loss = loss

        else:
            self.E += self.alpha_bias * self.bias
            np.random.seed(0)
            self.F = np.array(np.random.rand(int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
            self.F = normalize(self.F, axis=1, norm='l2')
            for step in range(self.steps_after):
                # print("iter = ", step,'\t', "Process 2 for training",'\t', (step * 100) / self.steps_after, "%")
                drv = np.zeros((int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
                drv = drv + 2 * alpha * np.dot(np.dot(self.E.T, self.E), self.F) - 2 * alpha * np.dot(self.E.T, self.graph) + 2 * self.lmbda * self.F
                rt = ((- drv.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                dt = rt
                drv.fill(0.0)
                vecF = (self.F.T).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                countF = 0
                while (True):
                    countF += 1
                    norm_rt = norm(rt)
                    if countF > 10:
                        break
                    else:
                        dtS = dt.reshape((dataset.n_ids, int(config.L_COMPONENTS))).T
                        storeHdt = 2 * alpha * np.dot(np.dot(self.E.T, self.E), dtS) + 2 * self.lmbda * dtS
                        Hdt = ((storeHdt.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                        rtrt = norm(rt) ** 2
                        dtHdt = np.sum(np.multiply(dt, Hdt))
                        at = rtrt / dtHdt
                        vecF = vecF + at * dt
                        rt = rt - at * Hdt
                        rtmprtmp = rtrt
                        rtrt += norm(rt) ** 2
                        bt = rtrt / rtmprtmp
                        dt = rt + bt * dt
                Ft = (vecF.flatten()).reshape((dataset.n_ids, int(config.L_COMPONENTS)))
                self.F = Ft.T

                EF = np.dot(self.E, self.F)
                fitMEF = norm(self.graph - EF) ** 2
                loss = alpha * fitMEF + (self.lmbda) * (norm(self.E) ** 2 + norm(self.F) ** 2)
                if (abs(last_loss - loss) < float(config.CONV_LS)):
                    break
                last_loss = loss


            np.random.seed(0)
            self.E = np.array(np.random.rand(dataset.n_ids, int(config.L_COMPONENTS)), dtype=np.float32)
            self.E = normalize(self.E, axis=1, norm='l2')
            for step in range(self.steps_after):
                # print("iter = ", step,'\t', "Process 2 for training",'\t', (step * 100) / self.steps_after, "%")
                B = np.copy(self.F)
                drv = np.zeros((int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
                Hess = np.zeros((int(config.L_COMPONENTS), int(config.L_COMPONENTS)), dtype=np.float32)
                drv = drv - 2 * alpha * np.dot(B, self.graph.T) + 2 * alpha * np.dot(np.dot(B, B.T),
                                                                     self.E.T) + 2 * self.lmbda * self.E.T
                Hess = Hess + 2 * alpha * np.dot(B, B.T)
                np.fill_diagonal(Hess, np.diagonal(Hess) + 2 * self.lmbda)
                rt = ((- drv.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                dt = rt
                drv.fill(0.0)
                vecE = (self.E.flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))

                # Update E
                countE = 0
                while (True):
                    countE += 1
                    norm_rt = norm(rt)
                    if countE > 10:
                        break
                    else:
                        storeHessdtS = np.zeros((int(config.L_COMPONENTS), dataset.n_ids), dtype=np.float32)
                        dtS = dt.reshape((dataset.n_ids, int(config.L_COMPONENTS))).T
                        storeHessdtS = storeHessdtS + np.dot(Hess, dtS)
                        Hdt = ((storeHessdtS.T).flatten()).reshape((int(config.L_COMPONENTS) * dataset.n_ids, 1))
                        rtrt = norm(rt) ** 2
                        dtHdt = np.sum(np.multiply(dt, Hdt))
                        at = rtrt / dtHdt
                        vecE = vecE + at * dt
                        rt = rt - at * Hdt
                        rtmprtmp = rtrt
                        rtrt += norm(rt) ** 2
                        bt = rtrt / rtmprtmp
                        dt = rt + bt * dt
                self.E = vecE.reshape((dataset.n_ids, int(config.L_COMPONENTS)))

                EF = np.dot(self.E, self.F)
                fitMEF = norm(self.graph - EF) ** 2
                loss = alpha * fitMEF + (self.lmbda) * (norm(self.E) ** 2 + norm(self.F) ** 2)
                if (abs(last_loss - loss) < float(config.CONV_LS)):
                    break
                last_loss = loss


    def run(self, config, dataset, logger, S, Y, train_ids, val_ids, test_ids):
        print("Experiment details : Perc: {%d}, Alpha Bias: {%0.5f}, Lambda : {%0.5f}, Alpha: {%0.5f}" % (self.flagnum, self.alpha_bias,  self.lmbda, self.alpha))
        loop_size = 10
        max_svm_accu = -1
        stop_index = int(config.STOP_INDEX)
        early_stopping = int(config.EARLY_STOPPING)
        for i in range(loop_size):
            tic = time.time()
            if (i == 0):
                self.make_graph(config, dataset, logger, S)
            else:
                self.train_w(config, dataset, logger)
            toc = time.time()
            # print(i, "\tTrain embeddings over! use time ", toc - tic )
            svm = evaluate_svm(config, dataset, logger, self.E, train_ids, val_ids, test_ids)
            saw, performance = svm.train_evaluate_svm(config, dataset, logger)
            self.saw = saw

            if ((i % config.STEP) == 0):
                from main_algo import get_perf_metrics
                # Can be set test_ids during testing or val_ids during hyper-param search using validation
                val_svm_accu = get_perf_metrics(config, self.E, self.saw.weight_b, Y, train_ids, test_ids, 'svm')
                if val_svm_accu["micro_f1"] > max_svm_accu:
                    best_svm_result = {'Q': self.saw.weight_b, 'U': self.E, 'i': i}
                    max_svm_accu = val_svm_accu["micro_f1"]
                    stop_index = 0
                else:
                    stop_index = stop_index + 20

            if stop_index > early_stopping:
                logger.debug("Early stopping")
                break

        return best_svm_result