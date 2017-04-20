'''
A script to handle the plotting of different ranking model performances
on a bar graph.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model_df(data_file, model, ignore_fold):
	'''
	Retrieves a dataframe of a model, including its evaluated metrics for all folds.

	Args:
        data_file (str): the data file the model data is stored in.
        model (str): the path file of the model we want to get the data frame of.
        ignore_fold (str): ignore any folds.

    Returns:
        DataFrame: a dataframe consisting evaluated metrics for all folds.
	'''
	df = pd.read_csv(data_file)
	df = df[df['model'] == model][df['fold'] != ignore_fold]

	return df

def retrieve_values(df):
	'''
	Args:
        df (DataFrame): .

    Returns:
        tuple: a tuple of attributes from the dataframe including
        	   ndcg_mean/min/max and map_mean/min/max
    '''
	ndcg_mean = np.mean(df['ndcg@10'].values)
	ndcg_min = min(df['ndcg@10'].values)
	ndcg_max = max(df['ndcg@10'].values)

	map_mean = np.mean(df['MAP'].values)
	map_min = min(df['MAP'].values)
	map_max = max(df['MAP'].values)

	return ndcg_mean, ndcg_min, ndcg_max, map_mean, map_min, map_max


def bar_attributes(data_file, model, ignore_fold):
	'''
	Retrieves the bar chart attributes of a model for all folds.

	Args:
        data_file (str): the data file the model data is stored in.
        model (str): the path file of the model we want to get the data frame of.
        ignore_fold (str): ignore any folds.

    Returns:
        tuple: a tuple consisting of evaluated metrics for all folds and error bars.
	'''
	df = get_model_df(data_file, model, ignore_fold)
	ndcg_mean, ndcg_min, ndcg_max, map_mean, map_min, map_max = retrieve_values(df)

	ndcg_err_min = ndcg_mean - ndcg_min
	ndcg_err_max = ndcg_max - ndcg_mean

	map_err_min = map_mean - map_min
	map_err_max = map_max - map_mean

	return ndcg_mean, ndcg_err_min, ndcg_err_max, map_mean, map_err_max, map_err_min

def autolabel(rects):
	    """
	    Attach a text label above each bar displaying its height
	    """
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                '%.2f' % float(height),
	                ha='center', va='bottom')


if __name__ == '__main__':
	models = [('metric_output/ranknet_metrics_output_all(exponential).txt', 'ranknet_lr0.000005.txt'),
	 		  ('metric_output/lambdaMART_metrics_output_all(exponential).txt', 'lambdaMART_leaves10.txt'),
	 		  ('metric_output/randomforests_metrics_output_all(exponential).txt', 'randomforests_leaf100.txt')]

	ind = np.arange(2)
	width = 0.20
	fig, ax = plt.subplots()
	rects = []

	for index, (data_file, model) in enumerate(models):
		if 'randomforests' in model:
			ignore_fold = 1
		else:
			ignore_fold = 4

		bar_attrs = bar_attributes(data_file, model, ignore_fold)

		means = (bar_attrs[0], bar_attrs[3])
		yerr = ([bar_attrs[1], bar_attrs[2]], [bar_attrs[4], bar_attrs[5]])
		
		rect = ax.bar(ind + (index+3)*width, means, width, color=['r', 'b', 'c'][index], yerr=yerr, error_kw=dict(ecolor='black', lw=2, capsize=4, capthick=2))
		rects.append(rect)

	baseline = (0.17993507552755053, 0.0060826471293821127, 0.0040572336045003876, 0.46607170904565631, 0.0061445879328302921, 0.0065974767329390893)
	means = (baseline[0], baseline[3])
	yerr = ([baseline[1], baseline[2]], [baseline[4], baseline[5]])
	rect_baseline = ax.bar(ind + 2*width, means, width, color='g', yerr=yerr, error_kw=dict(ecolor='black', lw=2, capsize=4, capthick=2))
	rects.append(rect_baseline)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Metric Scores')
	ax.set_title('Comparison of metric scores between different algorithms.')
	ax.set_xticks(ind + 8*width / 2)
	ax.set_xticklabels(('NDCG@10', 'MAP'))

	ax.legend(tuple(map(lambda x: x[0], rects)), ('RankNet', 'LambdaMART', 'RandomForests', 'Baseline'), loc=2)
	ax.set_ylim(0, 0.8)

	for rect in rects:
		autolabel(rect)

	plt.savefig("algorithm_comparisons_default.png")
	plt.show()
