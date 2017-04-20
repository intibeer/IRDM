'''
A file to handle the graphing of tuning parameters for RankNet.
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

	df = pd.read_csv('metric_output/ranknet_metrics_output_all(exponential)_binaryMAP.txt')

	list_of_models = df['model'].unique()

	nodes_ndcg_mean = []
	nodes_ndcg_min = []
	nodes_ndcg_max = []
	nodes_map_mean = []
	nodes_map_min = []
	nodes_map_max = []
	nodes_numbers = []

	layers_ndcg_mean = []
	layers_ndcg_min = []
	layers_ndcg_max = []
	layers_map_mean = []
	layers_map_min = []
	layers_map_max = []
	layers_numbers = []

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

		ndcg_min = min(dataframe['ndcg@10'])
		ndcg_mean = np.mean(dataframe['ndcg@10'])
		ndcg_max = max(dataframe['ndcg@10'])

		map_min = min(dataframe['MAP'])
		map_mean = np.mean(dataframe['MAP'])
		map_max = max(dataframe['MAP'])

		if("nodes" in dataframe['model'].values[0]):
			index_start = dataframe['model'].values[0].index("nodes")
			tree_number = dataframe['model'].values[0][index_start + len("nodes"):].replace('.txt', '')

			nodes_ndcg_mean.append(ndcg_mean)
			nodes_ndcg_max.append(ndcg_max - ndcg_mean)
			nodes_ndcg_min.append(ndcg_mean - ndcg_min)
			nodes_map_mean.append(map_mean)
			nodes_map_max.append(map_max - map_mean)
			nodes_map_min.append(map_mean - map_min)
			
			nodes_numbers.append(float(tree_number))
			
		elif("layers" in dataframe['model'].values[0]):
			index_start = dataframe['model'].values[0].index("layers")
			leaf_number = dataframe['model'].values[0][index_start + len("layers"):].replace('.txt', '')

			layers_ndcg_mean.append(ndcg_mean)
			layers_ndcg_max.append(ndcg_max - ndcg_mean)
			layers_ndcg_min.append(ndcg_mean - ndcg_min)
			layers_map_mean.append(map_mean)
			layers_map_max.append(map_max - map_mean)
			layers_map_min.append(map_mean - map_min)
			
			layers_numbers.append(float(leaf_number))

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

	d = {'ndcg@10_mean': pd.Series(nodes_ndcg_mean, index=nodes_numbers), 'ndcg@10_min': pd.Series(nodes_ndcg_min, index=nodes_numbers), 'ndcg@10_max': pd.Series(nodes_ndcg_max, index=nodes_numbers), 'map_mean': pd.Series(nodes_map_mean, index=nodes_numbers), 'map_min': pd.Series(nodes_map_min, index=nodes_numbers), 'map_max': pd.Series(nodes_map_max, index=nodes_numbers)}
	df_nodes = pd.DataFrame(d)

	d = {'ndcg@10_mean': pd.Series(layers_ndcg_mean, index=layers_numbers), 'ndcg@10_min': pd.Series(layers_ndcg_min, index=layers_numbers), 'ndcg@10_max': pd.Series(layers_ndcg_max, index=layers_numbers), 'map_mean': pd.Series(layers_map_mean, index=layers_numbers), 'map_min': pd.Series(layers_map_min, index=layers_numbers), 'map_max': pd.Series(layers_map_max, index=layers_numbers)}
	df_layers = pd.DataFrame(d).sort_index(0)

	d = {'ndcg@10_mean': pd.Series(lr_ndcg_mean, index=lr_numbers), 'ndcg@10_min': pd.Series(lr_ndcg_min, index=lr_numbers), 'ndcg@10_max': pd.Series(lr_ndcg_max, index=lr_numbers), 'map_mean': pd.Series(lr_map_mean, index=lr_numbers), 'map_min': pd.Series(lr_map_min, index=lr_numbers), 'map_max': pd.Series(lr_map_max, index=lr_numbers)}
	df_lr = pd.DataFrame(d)

	scatter_plot(df_nodes, "ndcg@10", "nodes")
	scatter_plot(df_nodes, "map", "nodes")
	plt.title("Ranknet - Tuning number of nodes")
	plt.legend(loc=2)
	plt.ylim(0, 1)
	plt.savefig("figures/ranknet_nodes_metrics.png")
	plt.show()

	scatter_plot(df_layers, "ndcg@10", "layers")
	scatter_plot(df_layers, "map", "layers")
	plt.title("Ranknet - Tuning number of layers")
	plt.legend(loc=2)
	plt.ylim(0, 1)
	plt.savefig("figures/ranknet_layers_metrics.png")
	plt.show()

	scatter_plot(df_lr, "ndcg@10", "lr")
	scatter_plot(df_lr, "map", "lr")
	plt.title("Ranknet - Tuning the learning rate")
	plt.xlabel(r'Learning rate ($\log_{10}(l/5)$)')
	plt.legend(loc=2)
	plt.ylim(0, 1)
	plt.savefig("figures/ranknet_lr_metrics.png")
	plt.show()
