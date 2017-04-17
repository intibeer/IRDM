import numpy as np
import matplotlib.pyplot as plt

rn_layers_1 = [0.1953, 0.1975, 0.1930]
rn_nodes_1 = [0.1955, 0.1920, 0.1911, 0.1920]
rn_lr_1 = [0.1738, 0.1743, 0.1934, 0.1964, 0.1839, 0.1959]

rn_layers_2 = [0.2019, 0.2037, 0.2040]
rn_nodes_2 = [0.2092, 0.2024, 0.2021, 0.2024]
rn_lr_2 = [0.1819, 0.1818, 0.1999, 0.2073, 0.1927, 0.2068]

rn_layers_3 = [0.1944, 0.1964, 0.1938]
rn_nodes_3 = [0.2027, 0.1941, 0.2022, 0.1941]
rn_lr_3 = [0.1837, 0.1840, 0.1977, 0.2024, 0.1979, 0.2002]

rn_layers_4 = [0.1965, 0.1998, 0.1996]
rn_nodes_4 = [0.2006, 0.1962, 0.1947, 0.1962]
rn_lr_4 = [0.1764, 0.1764, 0.1961, 0.2043, 0.1877, 0.2028]

rn_layers_5 = [0.1959, 0.2021, 0.1996]
rn_nodes_5 = [0.2044, 0.1992, 0.2020, 0.1992]
rn_lr_5 = [0.1840, 0.1847, 0.1996, 0.2054, 0.1939, 0.2034]

lambda_mart_trees_1 = [0.3354, 0.3658, 0.4297, 0.4713, 0.4749]
lambda_mart_leaves_1 = [0.4179, 0.4484, 0.4713, 0.5039, 0.5189]
lambda_mart_shr_1 = [0.4507, 0.4713, 0.4300, 0.3515, 0.3515]

lambda_mart_trees_2 = [0.3374, 0.3699, 0.4339, 0.4710, 0.4748]
lambda_mart_leaves_2 = [0.4202, 0.4482, 0.4710, 0.5024, 0.5147]
lambda_mart_shr_2 = [0.4485, 0.4710, 0.4345, 0.3559, 0.3559]

lambda_mart_trees_3 = [0.3433, 0.3744, 0.4258, 0.4445, 0.4467]
lambda_mart_leaves_3 = [0.4164, 0.4385, 0.4445, 0.4513, 0.4482]
lambda_mart_shr_3 = [0.4353, 0.4445, 0.4266, 0.3612, 0.3612]

lambda_mart_trees_4 = [0.3379, 0.3668, 0.4313, 0.4524, 0.4533]
lambda_mart_leaves_4 = [0.4204, 0.4432, 0.4524, 0.4585, 0.4594]
lambda_mart_shr_4 = [0.4358, 0.4524, 0.4296, 0.3557, 0.3557]

lambda_mart_trees_5 = [0.3439, 0.3777, 0.4399, 0.4856, 0.4856]
lambda_mart_leaves_5 = [0.4239, 0.4579, 0.4813, 0.5140, 0.5295]
lambda_mart_shr_5 = [0.4577, 0.4813, 0.4405, 0.3642, 0.3642]


rn_layers = [
				[rn_layers_1[0], rn_layers_2[0], rn_layers_3[0], rn_layers_4[0], rn_layers_5[0]],
				[rn_layers_1[1], rn_layers_2[1], rn_layers_3[1], rn_layers_4[1], rn_layers_5[1]],
				[rn_layers_1[2], rn_layers_2[2], rn_layers_3[2], rn_layers_4[2], rn_layers_5[2]]
			]


rn_nodes = [
				[rn_nodes_1[0], rn_nodes_2[0], rn_nodes_3[0], rn_nodes_4[0], rn_nodes_5[0]],
				[rn_nodes_1[1], rn_nodes_2[1], rn_nodes_3[1], rn_nodes_4[1], rn_nodes_5[1]],
				[rn_nodes_1[2], rn_nodes_2[2], rn_nodes_3[2], rn_nodes_4[2], rn_nodes_5[2]],
				[rn_nodes_1[3], rn_nodes_2[3], rn_nodes_3[3], rn_nodes_4[3], rn_nodes_5[3]]
		]

rn_lr = [
				[rn_lr_1[0], rn_lr_2[0], rn_lr_3[0], rn_lr_4[0], rn_lr_5[0]],
				[rn_lr_1[1], rn_lr_2[1], rn_lr_3[1], rn_lr_4[1], rn_lr_5[1]],
				[rn_lr_1[2], rn_lr_2[2], rn_lr_3[2], rn_lr_4[2], rn_lr_5[2]],
				[rn_lr_1[3], rn_lr_2[3], rn_lr_3[3], rn_lr_4[3], rn_lr_5[3]],
				[rn_lr_1[4], rn_lr_2[4], rn_lr_3[4], rn_lr_4[4], rn_lr_5[4]],
				[rn_lr_1[5], rn_lr_2[5], rn_lr_3[5], rn_lr_4[5], rn_lr_5[5]]
		
		]

lambda_mart_trees = [
				[lambda_mart_trees_1[0], lambda_mart_trees_2[0],  lambda_mart_trees_3[0],  lambda_mart_trees_4[0],  lambda_mart_trees_5[0]],
				[lambda_mart_trees_1[1], lambda_mart_trees_2[1],  lambda_mart_trees_3[1],  lambda_mart_trees_4[1],  lambda_mart_trees_5[1]],
				[lambda_mart_trees_1[2], lambda_mart_trees_2[2],  lambda_mart_trees_3[2],  lambda_mart_trees_4[2],  lambda_mart_trees_5[2]],
				[lambda_mart_trees_1[3], lambda_mart_trees_2[3],  lambda_mart_trees_3[3],  lambda_mart_trees_4[3],  lambda_mart_trees_5[3]],
				[lambda_mart_trees_1[4], lambda_mart_trees_2[4],  lambda_mart_trees_3[4],  lambda_mart_trees_4[4],  lambda_mart_trees_5[4]],
]

lambda_mart_leaves = [
				[lambda_mart_leaves_1[0], lambda_mart_leaves_2[0],  lambda_mart_leaves_3[0],  lambda_mart_leaves_4[0],  lambda_mart_leaves_5[0]],
				[lambda_mart_leaves_1[1], lambda_mart_leaves_2[1],  lambda_mart_leaves_3[1],  lambda_mart_leaves_4[1],  lambda_mart_leaves_5[1]],
				[lambda_mart_leaves_1[2], lambda_mart_leaves_2[2],  lambda_mart_leaves_3[2],  lambda_mart_leaves_4[2],  lambda_mart_leaves_5[2]],
				[lambda_mart_leaves_1[3], lambda_mart_leaves_2[3],  lambda_mart_leaves_3[3],  lambda_mart_leaves_4[3],  lambda_mart_leaves_5[3]],
				[lambda_mart_leaves_1[4], lambda_mart_leaves_2[4],  lambda_mart_leaves_3[4],  lambda_mart_leaves_4[4],  lambda_mart_leaves_5[4]],
]

lambda_mart_shr = [
				[lambda_mart_shr_1[0], lambda_mart_shr_2[0],  lambda_mart_shr_3[0],  lambda_mart_shr_4[0],  lambda_mart_shr_5[0]],
				[lambda_mart_shr_1[1], lambda_mart_shr_2[1],  lambda_mart_shr_3[1],  lambda_mart_shr_4[1],  lambda_mart_shr_5[1]],
				[lambda_mart_shr_1[2], lambda_mart_shr_2[2],  lambda_mart_shr_3[2],  lambda_mart_shr_4[2],  lambda_mart_shr_5[2]],
				[lambda_mart_shr_1[3], lambda_mart_shr_2[3],  lambda_mart_shr_3[3],  lambda_mart_shr_4[3],  lambda_mart_shr_5[3]],
				[lambda_mart_shr_1[4], lambda_mart_shr_2[4],  lambda_mart_shr_3[4],  lambda_mart_shr_4[4],  lambda_mart_shr_5[4]],
]

def mean_values(list_values):
	return [np.mean(list_values[i]) for i in range(len(list_values))]

def plot_lr(list_values):
	mean = [np.mean(np.array(list_values)[i]) for i in range(np.array(list_values).T[0].size)]
	yerr = [[np.mean(np.array(list_values)[i]) - min(np.array(list_values)[i]) for i in range(np.array(list_values).T[0].size)], [max(np.array(list_values)[i] - np.mean(np.array(list_values)[i])) for i in range(np.array(list_values).T[0].size)]]
	xvals = [0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005]
	x_axis = [np.log10(i / 5) for i in xvals]
	plt.plot(x_axis, mean)
	plt.errorbar(x_axis, mean, yerr=yerr)
	plt.xlim(min(x_axis) - 1, max(x_axis) + 1)
	plt.xlabel("Learning Rate (log10(x/5))")
	plt.ylabel("NDCG")
	plt.show()

def plot_attr(list_values):
	print(np.array(list_values).T)
	mean = [np.mean(np.array(list_values)[i]) for i in range(np.array(list_values).T[0].size)]
	yerr = [[np.mean(np.array(list_values)[i]) - min(np.array(list_values)[i]) for i in range(np.array(list_values).T[0].size)], [max(np.array(list_values)[i] - np.mean(np.array(list_values)[i])) for i in range(np.array(list_values).T[0].size)]]
	print(yerr)
	xvals = [i for i in range(len(mean))]
	plt.plot(xvals, mean)
	plt.errorbar(xvals, mean, yerr=yerr)
	plt.xlim(min(xvals) - 1, max(xvals) + 1)
	plt.show()

plot_attr(lambda_mart_trees)
plot_attr(lambda_mart_leaves)
plot_attr(lambda_mart_shr)
plot_lr(rn_lr)