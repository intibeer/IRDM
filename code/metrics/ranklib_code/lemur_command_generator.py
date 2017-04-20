'''
Methods to generate RankLib command line commands.
'''

def generate_command_lambda_mart(fold_number, method_number, lr=None, trees=None, leaves=None):
	fold_number = str(fold_number)
	method_number = str(method_number)
	if(lr != None):
		lr = str(lr)
		return "nohup java -jar ../../RankLib-2.1-patched.jar -train ../../MSLR-WEB10K/Fold" + fold_number + "/train.txt -test ../../MSLR-WEB10K/Fold" + fold_number + "/test.txt -validate ../../MSLR-WEB10K/Fold" + fold_number + "/vali.txt -ranker " + method_number + " -shrinkage " + '{:.8f}'.format(float(lr)) + " -metric2t NDCG@10 -metric2T ERR@10 -save lambdaMART_fold" + fold_number + "_shr" + '{:.8f}'.format(float(lr)) + ".txt &"

	elif(trees != None):
		trees = str(trees)
		return "nohup java -jar ../../RankLib-2.1-patched.jar -train ../../MSLR-WEB10K/Fold" + fold_number + "/train.txt -test ../../MSLR-WEB10K/Fold" + fold_number + "/test.txt -validate ../../MSLR-WEB10K/Fold" + fold_number + "/vali.txt -ranker " + method_number + " -tree " + str(trees) + " -metric2t NDCG@10 -metric2T ERR@10 -save lambdaMART_fold" + fold_number + "_trees" + trees + ".txt &"		

	elif(leaves != None):
		leaves = str(leaves)
		return "nohup java -jar ../../RankLib-2.1-patched.jar -train ../../MSLR-WEB10K/Fold" + fold_number + "/train.txt -test ../../MSLR-WEB10K/Fold" + fold_number + "/test.txt -validate ../../MSLR-WEB10K/Fold" + fold_number + "/vali.txt -ranker " + method_number + " -leaf " + str(leaves) + " -metric2t NDCG@10 -metric2T ERR@10 -save lambdaMART_fold" + fold_number + "_leaves" + leaves + ".txt &"

def print_commands_lambdamart(fold_num):
	for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
		print(generate_command_lambda_mart(fold_num, 6, lr=lr) + "\n")

	for trees in [1, 10, 100, 1000, 10000]:
		print(generate_command_lambda_mart(fold_num, 6, trees=trees) + "\n")

	for leaves in [1, 5, 10, 50, 100, 500]:
		print(generate_command_lambda_mart(fold_num, 6, leaves=leaves) + "\n")


def generate_command_ranknet(fold_number, method_number, lr=None, nodes=None, layers=None):
	fold_number = str(fold_number)
	method_number = str(method_number)

	if(lr != None):
		lr = str(lr)
		return "nohup java -jar ../../RankLib-2.1-patched.jar -train ../../MSLR-WEB10K/Fold" + fold_number + "/train.txt -test ../../MSLR-WEB10K/Fold" + fold_number + "/test.txt -validate ../../MSLR-WEB10K/Fold" + fold_number + "/vali.txt -ranker " + method_number + " -lr " + '{:.8f}'.format(float(lr)) + " -metric2T ERR@10 -save ranknet_fold" + fold_number + "_lr" + '{:.8f}'.format(float(lr)) + ".txt &"

	elif(nodes != None):
		nodes = str(nodes)
		return "nohup java -jar ../../RankLib-2.1-patched.jar -train ../../MSLR-WEB10K/Fold" + fold_number + "/train.txt -test ../../MSLR-WEB10K/Fold" + fold_number + "/test.txt -validate ../../MSLR-WEB10K/Fold" + fold_number + "/vali.txt -ranker " + method_number + " -node " + str(nodes) + " -metric2T ERR@10 -save ranknet" + fold_number + "_nodes" + nodes + ".txt &"		

	elif(layers != None):
		layers = str(layers)
		return "nohup java -jar ../../RankLib-2.1-patched.jar -train ../../MSLR-WEB10K/Fold" + fold_number + "/train.txt -test ../../MSLR-WEB10K/Fold" + fold_number + "/test.txt -validate ../../MSLR-WEB10K/Fold" + fold_number + "/vali.txt -ranker " + method_number + " -layer " + str(layers) + " -metric2T ERR@10 -save ranknet" + fold_number + "_layers" + layers + ".txt &"

def print_commands_ranknet(fold_num):
	for lr in [0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.5]:
		print(generate_command_ranknet(fold_num, 1, lr=lr) + "\n")

	for nodes in [5, 10, 15, 20]:
		print(generate_command_ranknet(fold_num, 1, nodes=nodes) + "\n")

	for layers in [1, 2, 3]:
		print(generate_command_ranknet(fold_num, 1, layers=layers) + "\n")

print_commands_ranknet(fold_num=4)
print_commands_lambdamart(fold_num=4)