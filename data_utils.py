from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise
import networkx as nx

def get_all_label_distribution(truth):
    all_label_dist = np.true_divide(np.sum(truth, axis=0), np.sum(truth))
    return all_label_dist

def get_proximity_matrix(X, eta, nodelist=None):
    G = nx.from_numpy_matrix(X)
    if nodelist is None:
        nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, format='csr')
    n, m = M.shape
    DI = np.diagflat(1.0 / np.sum(M, axis=1))
    P = DI * M
    A = (P + eta * np.dot(P, P)) / 2
    return np.array(A).reshape((n, n))

def get_proximity_similarity_matrix(X, eta):
    S = (X + eta * pairwise.cosine_similarity(X)) / 2
    S = normalize(S, axis=1, norm='l2')
    return S

