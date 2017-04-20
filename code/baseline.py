'''
Retrieves the baseline metrics for NDCG@10 and MAP.
'''

from extract_relevancies import get_tagged_fold_relevancies
from metric import *

ndcg_baselines = []
map_baselines = []

for fold in [1, 2, 3, 4, 5]:
	fold_relevancies = get_tagged_fold_relevancies("../../MSLR-WEB10K/Fold" + str(fold) + "/test.txt")
	ndcg_baselines.append(np.mean([ndcg(map(lambda x: x[0], fold_relevancies[key]), 10, 'exp') for key in fold_relevancies]))
	map_baselines.append(mean_average_precision([map(lambda x: x[0], fold_relevancies[key]) for key in fold_relevancies], rel_con=1))


print(ndcg_baselines)
print(map_baselines)
print(np.mean(ndcg_baselines),np.mean(ndcg_baselines) - min(ndcg_baselines), max(ndcg_baselines) -np.mean(ndcg_baselines),np.mean(map_baselines),np.mean(map_baselines) - min(map_baselines), max(map_baselines) -np.mean(map_baselines))