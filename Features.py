import numpy as np

class Features:
    def __init__(self, rel, qid, features, index = None):
        self.rel = rel
        self.qid = qid
        self.features = np.array([features])
        self.index = index
        
    