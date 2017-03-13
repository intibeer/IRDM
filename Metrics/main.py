from metric import *

def parse_data(filename):
    query_data_pair_list = []
    seen_before = []
    dic = {}

    with open(filename, 'r') as f:
        for line in f:
            relevant_data = line.split(" ")[:2]

            relevance = int(relevant_data[0])
            qid = relevant_data[1].replace('qid:', '')

            if(qid in seen_before):
                dic[qid].add_relevance(relevance)
            else:
                query = Query(qid)
                dic[qid] = query
                seen_before.append(qid)

            query_data_pair_list.append(relevant_data)

    return list(dic.values())

query_array = parse_data("../MSLR-WEB10K/Fold1/test.txt")

for query in query_array:
    print(Metric(query).ndcg())
