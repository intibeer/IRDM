import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

df = pd.read_csv('metric_output/lambdaMART_metrics_output.txt')

list_of_models = df['model'].unique()

trees_ndcg_mean = []
trees_ndcg_min = []
trees_ndcg_max = []
trees_map_mean = []
trees_map_min = []
trees_map_max = []
trees_numbers = []

leaves_ndcg_mean = []
leaves_ndcg_min = []
leaves_ndcg_max = []
leaves_map_mean = []
leaves_map_min = []
leaves_map_max = []
leaves_numbers = []

shr_ndcg_mean = []
shr_ndcg_min = []
shr_ndcg_max = []
shr_map_mean = []
shr_map_min = []
shr_map_max = []
shr_numbers = []


for model in list_of_models:
	dataframe = df[df.model == model]

	ndcg_min = min(dataframe['ndcg@10'])
	ndcg_mean = np.mean(dataframe['ndcg@10'])
	ndcg_max = max(dataframe['ndcg@10'])

	map_min = min(dataframe['MAP'])
	map_mean = np.mean(dataframe['MAP'])
	map_max = max(dataframe['MAP'])

	if("trees" in dataframe['model'].values[0]):
		index_start = dataframe['model'].values[0].index("trees")
		tree_number = dataframe['model'].values[0][index_start + len("trees"):].replace('.txt', '')

		trees_ndcg_mean.append(ndcg_mean)
		trees_ndcg_max.append(ndcg_max - ndcg_mean)
		trees_ndcg_min.append(ndcg_mean - ndcg_min)
		trees_map_mean.append(map_mean)
		trees_map_max.append(map_max - map_mean)
		trees_map_min.append(map_mean - map_min)
		
		trees_numbers.append(float(tree_number))
		
	elif("leaves" in dataframe['model'].values[0]):
		index_start = dataframe['model'].values[0].index("leaves")
		leaf_number = dataframe['model'].values[0][index_start + len("leaves"):].replace('.txt', '')

		leaves_ndcg_mean.append(ndcg_mean)
		leaves_ndcg_max.append(ndcg_max - ndcg_mean)
		leaves_ndcg_min.append(ndcg_mean - ndcg_min)
		leaves_map_mean.append(map_mean)
		leaves_map_max.append(map_max - map_mean)
		leaves_map_min.append(map_mean - map_min)
		
		leaves_numbers.append(float(leaf_number))

	elif("shr" in dataframe['model'].values[0]):
		index_start = dataframe['model'].values[0].index("shr")
		shr_number = dataframe['model'].values[0][index_start + len("shr"):].replace('.txt', '')

		shr_ndcg_mean.append(ndcg_mean)
		shr_ndcg_max.append(ndcg_max - ndcg_mean)
		shr_ndcg_min.append(ndcg_mean - ndcg_min)
		shr_map_mean.append(map_mean)
		shr_map_max.append(map_max - map_mean)
		shr_map_min.append(map_mean - map_min)
		
		shr_numbers.append(float(shr_number))

d = {'ndcg@10_mean': pd.Series(trees_ndcg_mean, index=trees_numbers), 'ndcg@10_min': pd.Series(trees_ndcg_min, index=trees_numbers), 'ndcg@10_max': pd.Series(trees_ndcg_max, index=trees_numbers), 'map_mean': pd.Series(trees_map_mean, index=trees_numbers), 'map_min': pd.Series(trees_map_min, index=trees_numbers), 'map_max': pd.Series(trees_map_max, index=trees_numbers)}
df_trees = pd.DataFrame(d)

d = {'ndcg@10_mean': pd.Series(leaves_ndcg_mean, index=leaves_numbers), 'ndcg@10_min': pd.Series(leaves_ndcg_min, index=leaves_numbers), 'ndcg@10_max': pd.Series(leaves_ndcg_max, index=leaves_numbers), 'map_mean': pd.Series(leaves_map_mean, index=leaves_numbers), 'map_min': pd.Series(leaves_map_min, index=leaves_numbers), 'map_max': pd.Series(leaves_map_max, index=leaves_numbers)}
df_leaves = pd.DataFrame(d).sort_index(0)

d = {'ndcg@10_mean': pd.Series(shr_ndcg_mean, index=shr_numbers), 'ndcg@10_min': pd.Series(shr_ndcg_min, index=shr_numbers), 'ndcg@10_max': pd.Series(shr_ndcg_max, index=shr_numbers), 'map_mean': pd.Series(shr_map_mean, index=shr_numbers), 'map_min': pd.Series(shr_map_min, index=shr_numbers), 'map_max': pd.Series(shr_map_max, index=shr_numbers)}
df_shr = pd.DataFrame(d)


def scatter_plot(df, metric, name):
	yerr_min = df[metric + "_min"].values
	yerr_max = df[metric + "_max"].values

	mean = df[metric + "_mean"]
	x = df.index.tolist()
	plt.scatter(x, mean, label=None)
	plt.errorbar(x, mean, yerr=[yerr_min, yerr_max], label=metric.upper())
	plt.xlabel("Number of " + name)
	plt.ylabel("Metric")

scatter_plot(df_trees, "ndcg@10", "trees")
scatter_plot(df_trees, "map", "trees")
plt.title("LambdaMART - Tuning number of trees")
plt.legend(loc=4)
plt.savefig("lambdaMART_trees_metrics.png")

scatter_plot(df_leaves, "ndcg@10", "leaves")
scatter_plot(df_leaves, "map", "leaves")
plt.title("LambdaMART - Tuning number of leaves")
plt.legend(loc=4)
plt.savefig("lambdaMART_leaves_metrics.png")

scatter_plot(df_shr, "ndcg@10", "shr")
scatter_plot(df_shr, "map", "shr")
plt.title("LambdaMART - Tuning the shrinkage (learning rate)")
plt.legend(loc=4)
plt.savefig("lambdaMART_shrinkage_metrics.png")