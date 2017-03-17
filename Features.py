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
    """
    Cost function for 2 feature objects
    INPUT:
        features_1: Feature object
            The first feature object
        features_2: Feature object
            The second feature object
        sigma: double
            Arbitrary sigma value which determines the shape of the sigmoid
    OUTPUT:
        cost: double
            Calculated cost value between features 1 and 2
    """
    s_i = features_1.rel
    s_j = features_2.rel
    s_ij = find_s_ij(s_i, s_j)
    return 0.5*(1-s_ij)*sigma*(s_i-s_j) + np.log10(1+np.exp(-sigma*(s_i-s_j)))
    
def cost_gradient(features_1, features_2, sigma):
    """
    Gradtient of cost function with respect to the first feature object
    INPUT:
        features_1: Feature object
            The first feature object
        features_2: Feature object
            The second feature object
        sigma: double
            Arbitrary sigma value which determines the shape of the sigmoid
    OUTPUT:
        cost gradient: double
            Calculated cost gradient value with respect to the first feature
            object
    """
    s_i = features_1.rel
    s_j = features_2.rel
    s_ij = find_s_ij(s_i, s_j)
    return sigma*(0.5*(1-s_ij)-1/(1+np.exp(sigma*(s_i-s_j))))