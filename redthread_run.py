import pickle as pkl
import numpy as np
from redthread import RedThread
from argparse import ArgumentParser

def get_args():
	parser = ArgumentParser()
	parser.add_argument("-data","--data_file",default="./sample_data/sampled_data_features.pkl", help="path to the data pickle files containing the freature matrix")
	parser.add_argument("-label","--label_file", default="./sample_data/sampled_data_labels.pkl", help='path to the labels of the data points as a pickle file')
	parser.add_argument("-modality", "--feature_name_file", default="./sample_data/sampled_data_feature_names.pkl", help="path to the feature names of the data as as pickle file")
	parser.add_argument("--data_folder", default="./sample_data/", help="path to the folder containing the different feature files")
	parser.add_argument("--budget", default=100, type=int, help='Number of times the model can query the user')
	parser.add_argument("--build_graph", default=False, type=bool, help='True if you want to build the data graph from scratch and False to use the pre-built graph')
	args = parser.parse_args()
	return args

def iterative_labelling(seed, budget, rt):
	query_counter = 0 # initializing the query counter "b" in the algorithm
	label_hash = rt.label_hash
	rt_graph = rt.get_graph()
	picked_nodes = []
	precision = 0.
	recall = 0.
	num_relevant = rt.num_positive_labels()
	 
	while query_counter < budget:
		print("Remaining Number of queries : " + str(budget-query_counter))
		picked_node = rt.infer_redthread() # pick a data point
		if rt.near_duplicate(picked_node): # check if the picked node is a near dupliate of the positively labelled nodes so far
			continue
	
		if picked_node not in list(label_hash.keys()):
			picked_node_label = rt.oracle(picked_node) # querying the user for the label of the picked node
			rt.update_label_hash(picked_node, picked_node_label) # update the label hash based on the oracle output
			query_counter += 1 
			if picked_node_label == 1:
				precision += 1
				recall += 1
		else:
			picked_node_label = label_hash[picked_node]
		rt.update_redthread(picked_node, picked_node_label)
		picked_nodes.append(picked_node)
		
		#print(label_hash)
	precision /= budget
	recall /= num_relevant
	return precision, recall

# def word_to_id(words):
# 	# assigns an id to each of the feature words
# 	word_id_map = {}
# 	counter = 1
# 	for word in set(words):
# 		word_id_map[word] = -counter
# 		counter += 1
# 	return word_id_map

# def get_word_id(words, word_id_map):
# 	# stores the id of the word in the feature_map
# 	word_ids = []
# 	for word in set(words):
# 		word_ids.append(word_id_map[word])
# 	return word_ids

def extract_info(args):
	data_file = args.data_file
	label_file = args.label_file
	all_feature_file = args.feature_name_file

	data = pkl.load(open(data_file, "rb"))
	labels = pkl.load(open(label_file, "rb"))
	feature_names = pkl.load(open(all_feature_file, "rb"))
	feature_map = {}
	feature_map["desc_uni"] = pkl.load(open(args.data_folder + "desc_feature_names_uni.pkl","rb"))
	feature_map["desc_bi"] = pkl.load(open(args.data_folder + "desc_feature_names_bi.pkl","rb"))
	feature_map["title_uni"] = pkl.load(open(args.data_folder + "title_feature_names_uni.pkl","rb"))
	feature_map["title_bi"] = pkl.load(open(args.data_folder + "title_feature_names_bi.pkl","rb"))
	feature_map["loc_uni"] = pkl.load(open(args.data_folder + "loc_feature_names.pkl","rb"))

	return data, labels, feature_names, feature_map

if __name__ == "__main__":

	# get command line arguments
	args = get_args()

	# get the data and labels and feature names
	data, labels, feature_names, feature_map = extract_info(args)

	# choosing random seed nodes for now
	total_prec = 0.
	total_rec = 0.

	seeds = np.random.choice(len(data), size=10, replace=False)

	for seed in seeds:
		# create a RedThread object
		rt = RedThread(data, labels, seed, feature_names, feature_map, args.build_graph)

		precision, recall = iterative_labelling(seed, args.budget, rt)
		f1_score = 2 * precision * recall / (precision + recall)
		print("Precision = " + str(precision))
		print("Recall = " + str(recall))
		print("F1 score = " + str(f1_score))
		total_prec += precision
		total_rec += recall

	total_prec /= len(seeds)
	total_rec /= len(seeds)
	total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec)
	print("Total precision = " + str(total_prec))
	print("Total recall = " + str(total_rec))
	print("Total f1 score = " + str(total_f1))
	print("success")


'''
Precision = 0.6
Recall = 0.024
F1 score = 0.04615384615384615

Precision = 0.65
Recall = 0.026
F1 score = 0.04999999999999999

Precision = 0.7
Recall = 0.028
F1 score = 0.05384615384615385

'''