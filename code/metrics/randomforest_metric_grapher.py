'''
A file to handle the graphing of tuning parameters for Random Forests.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(df, metric, name):
	'''
    Plots scatter points on a graph for a given parameter and metric.

    Args:
    	df (DataFrame): the model DataFrame.
    	metric (str): the metric to consider.
    	name (str): the parameter to plot.

    Returns:
        None
    '''
	yerr_min = df[metric + "_min"].values
	yerr_max = df[metric + "_max"].values

	mean = df[metric + "_mean"]
	x = df.index.tolist()
	plt.scatter(x, mean, label=None)
	plt.axhline(y=mean_baseline_ndcg, xmin=0, xmax=10000000, hold=None, color='k', alpha=0.5)
	plt.axhline(y=mean_baseline_map, xmin=0, xmax=10000000, hold=None, color='k', alpha=0.5)
	plt.errorbar(x, mean, yerr=[yerr_min, yerr_max], label=metric.upper())
	plt.xlabel("Number of " + name)
	plt.ylabel("Metric")

if __name__ == "__main__":
	df = pd.read_csv('metric_output/randomforests_metrics_output_all(exponential).txt')

	list_of_models = df['model'].unique()

	leaf_ndcg_mean = []
	leaf_ndcg_min = []
	leaf_ndcg_max = []
	leaf_map_mean = []
	leaf_map_min = []
	leaf_map_max = []
	leaf_numbers = []

	bags_ndcg_mean = []
	bags_ndcg_min = []
	bags_ndcg_max = []
	bags_map_mean = []
	bags_map_min = []
	bags_map_max = []
	bags_numbers = []

	lr_ndcg_mean = []
	lr_ndcg_min = []
	lr_ndcg_max = []
	lr_map_mean = []
	lr_map_min = []
	lr_map_max = []
	lr_numbers = []

	ndcg_baseline = [0.17385242839816836, 0.18181859425621524, 0.18363029558722849, 0.1763817502640895, 0.18399230913205075]
	map_baseline = [0.46500409887628907, 0.46750883515703351, 0.4599271211128253, 0.465249304303539, 0.47266918577859518]
	mean_baseline_ndcg = np.mean(ndcg_baseline)
	mean_baseline_map = np.mean(map_baseline)

	for model in list_of_models:
		dataframe = df[df.model == model]
		dataframe = dataframe[dataframe['fold'] != 1] # do something about this (ignores trained fold	)

		ndcg_min = min(dataframe['ndcg@10'])
		ndcg_mean = np.mean(dataframe['ndcg@10'])
		ndcg_max = max(dataframe['ndcg@10'])

		map_min = min(dataframe['MAP'])
		map_mean = np.mean(dataframe['MAP'])
		map_max = max(dataframe['MAP'])


		if("leaf" in dataframe['model'].values[0]):
			index_start = dataframe['model'].values[0].index("leaf")
			tree_number = dataframe['model'].values[0][index_start + len("leaf"):].replace('.txt', '')

			leaf_ndcg_mean.append(ndcg_mean)
			leaf_ndcg_max.append(ndcg_max - ndcg_mean)
			leaf_ndcg_min.append(ndcg_mean - ndcg_min)
			leaf_map_mean.append(map_mean)
			leaf_map_max.append(map_max - map_mean)
			leaf_map_min.append(map_mean - map_min)
			
			leaf_numbers.append(float(tree_number))
			
		elif("bags" in dataframe['model'].values[0]):
			index_start = dataframe['model'].values[0].index("bags")
			leaf_number = dataframe['model'].values[0][index_start + len("bags"):].replace('.txt', '')

			bags_ndcg_mean.append(ndcg_mean)
			bags_ndcg_max.append(ndcg_max - ndcg_mean)
			bags_ndcg_min.append(ndcg_mean - ndcg_min)
			bags_map_mean.append(map_mean)
			bags_map_max.append(map_max - map_mean)
			bags_map_min.append(map_mean - map_min)
			
			bags_numbers.append(float(leaf_number))

		elif("lr" in dataframe['model'].values[0]):
			index_start = dataframe['model'].values[0].index("lr")
			lr_number = dataframe['model'].values[0][index_start + len("lr"):].replace('.txt', '')

			lr_ndcg_mean.append(ndcg_mean)
			lr_ndcg_max.append(ndcg_max - ndcg_mean)
			lr_ndcg_min.append(ndcg_mean - ndcg_min)
			lr_map_mean.append(map_mean)
			lr_map_max.append(map_max - map_mean)
			lr_map_min.append(map_mean - map_min)
			
			lr_numbers.append(float(lr_number))

	d = {'ndcg@10_mean': pd.Series(leaf_ndcg_mean, index=leaf_numbers), 'ndcg@10_min': pd.Series(leaf_ndcg_min, index=leaf_numbers), 'ndcg@10_max': pd.Series(leaf_ndcg_max, index=leaf_numbers), 'map_mean': pd.Series(leaf_map_mean, index=leaf_numbers), 'map_min': pd.Series(leaf_map_min, index=leaf_numbers), 'map_max': pd.Series(leaf_map_max, index=leaf_numbers)}
	df_leaf = pd.DataFrame(d).sort_index(0)

	d = {'ndcg@10_mean': pd.Series(bags_ndcg_mean, index=bags_numbers), 'ndcg@10_min': pd.Series(bags_ndcg_min, index=bags_numbers), 'ndcg@10_max': pd.Series(bags_ndcg_max, index=bags_numbers), 'map_mean': pd.Series(bags_map_mean, index=bags_numbers), 'map_min': pd.Series(bags_map_min, index=bags_numbers), 'map_max': pd.Series(bags_map_max, index=bags_numbers)}
	df_bags = pd.DataFrame(d).sort_index(0)

	d = {'ndcg@10_mean': pd.Series(lr_ndcg_mean, index=lr_numbers), 'ndcg@10_min': pd.Series(lr_ndcg_min, index=lr_numbers), 'ndcg@10_max': pd.Series(lr_ndcg_max, index=lr_numbers), 'map_mean': pd.Series(lr_map_mean, index=lr_numbers), 'map_min': pd.Series(lr_map_min, index=lr_numbers), 'map_max': pd.Series(lr_map_max, index=lr_numbers)}
	df_lr = pd.DataFrame(d)

	scatter_plot(df_leaf, "ndcg@10", "leaf")
	scatter_plot(df_leaf, "map", "leaf")
	plt.xlabel("Number of leaves")
	plt.title("Random Forests Leaf Tuning")
	plt.legend(loc=2)
	plt.ylim(0, 1)
	plt.savefig("randomforests_leaf_metrics.png")
	plt.show()

	scatter_plot(df_bags, "ndcg@10", "bags")
	scatter_plot(df_bags, "map", "bags")
	plt.xlabel("Number of bags")
	plt.title("Random Forests Bags Tuning")
	plt.legend(loc=2)
	plt.ylim(0, 1)
	plt.savefig("randomforests_bags_metrics.png")
	plt.show()