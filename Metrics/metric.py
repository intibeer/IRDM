import numpy as np
from query import Query


class Metric(object):
    def __init__(self, query):
        self.query = query

    def cg(self):
        return sum(query.relevance)

    def dcg(self):
        summation = 0
        sorted_relevance = sorted(self.query.relevance)

        for index, rel in enumerate(sorted_relevance):
            summation += float(rel) / np.log2(index + 1 + 1)

        return summation

    def idcg(self):
        summation = 0
        sorted_relevance = sorted(self.query.relevance)

        for index, rel in enumerate(sorted_relevance):
            summation += (2**(rel) - 1)/np.log2(index + 1 + 1)

        return summation

    def ndcg(self):
        try:
            ndcg_value = self.dcg() / self.idcg()
        
        except ZeroDivisionError:
            ndcg_value = 0

        if(np.isnan(ndcg_value)):
            return -1
        else:
            return ndcg_value
