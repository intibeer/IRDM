import numpy as np
import matplotlib.pyplot as plt

num_of_folds = 5

ranknet_ndcg = (0.1839, 0.1927, 0.1979, 0.1877, 0.1939)
lambdamart_ndcg = (0.4713, 0.4710, 0.4445, 0.4524, 0.4813)
baseline_ndcg = (0.1739, 0.1818, 0.1836, 0.1764, 0.1840)

ind = np.arange(num_of_folds)
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(ind, baseline_ndcg, width, color='r')
rects2 = ax.bar(ind + width, ranknet_ndcg, width, color='b')
rects3 = ax.bar(ind + 2 * width, lambdamart_ndcg, width, color='g')

ax.set_ylabel('NDCG@10 Metric')
ax.set_title('NDCG@10 by algorithm and fold')
ax.set_xticks(ind + 3 * width / 2)
ax.set_xticklabels(('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Baseline', 'Ranknet', 'LambdaMART'))

plt.show()