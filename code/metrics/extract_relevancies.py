import numpy as np
from metric import *
import sys
import glob
import re
#print(documents)

def unique_document_ids(documents):
	dic = {}

	for document in documents:
		dic[document[0]] = int(document[0])

	return sorted(dic.values())

def get_untagged_ordered_relevancies(filename, fold_relevancies, number_of_queries_considered = 100):
	documents = []
	queries = []

	with open(filename, 'r') as f:
		for line in f.readlines():
			documents.append(line.split())

	document_ids = unique_document_ids(documents)

	for idd in document_ids[:number_of_queries_considered]:
		filtered_documents = filter(lambda document: document[0] == str(idd), documents)
		relevancy_list = map(lambda document: (document[1], float(document[2])), filtered_documents)
		relevancy_ordered_list = sorted(relevancy_list, key=lambda x: x[1], reverse=True)
		
		query = Query(idd, relevancy_ordered_list, fold_relevancies)
		queries.append(query)

	return queries

def get_tagged_fold_relevancies(fold_path):
	with open(fold_path, 'r') as f:
		qid = -1
		dic = {}
		counter = 0

		for line in f.readlines():
			document_relevancy = line.split()[:2]
			if(document_relevancy[1][4:] == qid):
				counter += 1
				dic[qid].append((int(document_relevancy[0]), counter))

			else:
				qid = document_relevancy[1][4:]

				counter = 0
				dic[qid] = [(int(document_relevancy[0]), counter)]

	return dic

class Query: 
	def __init__(self, qid, relevancies, fold_relevancies):
		self.qid = qid
		self.relevancies = relevancies
		self.fold_relevancies = fold_relevancies

	def get_previous_relevancies(self):
		return self.fold_relevancies[str(self.qid)]

	def order_relevancies(self):
		relevancies = []
		document_order = map(lambda x: int(x[0]), self.relevancies)

		for document_id in document_order:
			relevancies.append(max(map(lambda x: x[0] if x[1] == document_id else -1, self.get_previous_relevancies())))
		
		return relevancies


if __name__ == '__main__':
	fold_dir = sys.argv[1]
	model_dir = sys.argv[2]
	model_type = sys.argv[3]

	folds = glob.glob(fold_dir + "/*")
	models = glob.glob(model_dir + "/" + model_type + "/output_fold_*")

	for fold_num in range(len(folds)):
		for model_file in glob.glob(models[fold_num] + "/*"):
			fold_relevancies = get_tagged_fold_relevancies(folds[fold_num] + "/test.txt")

			map_relevancies = []
			ndcg_list = []

			for query in get_untagged_ordered_relevancies(model_file, fold_relevancies):
				ndcg_list.append(ndcg(query.order_relevancies(), 10))
				map_relevancies.append(query.order_relevancies())

			mean_ndcg = np.mean(ndcg_list)
			mean_avg_precision = mean_average_precision(map_relevancies, rel_con = 1)

			with open("metric_output/" + model_type + "_metrics_output.txt", 'a') as f:
				f.write(str(fold_num + 1) + ", " + str(model_file.split('/')[-1]) + ", " + str(mean_ndcg) + ", " + str(mean_avg_precision) + "\n")


