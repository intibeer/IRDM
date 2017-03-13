class Query(object):
    def __init__(self, qid):
        self.qid = qid
        self.relevance = []

    def add_relevance(self, value):
        self.relevance.append(value)
