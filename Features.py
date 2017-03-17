import numpy as np

class Features:
    def __init__(self, rel, qid, features, index = None):
        self.rel = rel
        self.qid = qid
        self.features = np.array([features])
        self.index = index
        
def find_s_ij(s_i, s_j):
    if s_i > s_j:
        return 1
    elif s_i == s_j:
        return 0
    else:
        return -1
        
def cost(features_1, features_2, sigma):
    s_i = features_1.rel
    s_j = features_2.rel
    s_ij = find_s_ij(s_i, s_j)
    return 0.5*(1-s_ij)*sigma*(s_i-s_j) + np.log10(1+np.exp(-sigma*(s_i-s_j)))