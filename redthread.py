import numpy as np
import os
import networkx as nx
import queue as Q
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix
from tqdm import tqdm


class RedThread:
	def __init__(self, labels, seed, feature_names, feature_map, queue_size=100, lr=0.1):
		# constructor for the class
		# self.data = csr_matrix(data)
		self.labels = labels
		self.label_hash = {seed:1}
		# self.rev_label_hash = {1:[seed],-1:[]} # keeps track of which nodes have positive and negative labels
		# self.num_features = data.shape[1]
		# self.related_nodes = {}
		self.modality_weight = {}
		# self.feature_names = feature_names
		# self.feature_map = feature_map
		num_feature_map = len(list(feature_map.keys()))
		self.queue_size = queue_size
		self.learning_rate = lr
		self.redthread_q = Q.PriorityQueue(maxsize=queue_size) # creating a priority queue to find the next inferred node (which has max weighted evidence flow)
		self.nodes_in_q = {}
		self.shell = {}
		self.evidence_flow_map = {}
		#self.initialize_related_nodes()
		# degrees = dict(graph.degree())
		# self.find_neighbours(graph)
		# self.neighbors = self.get_neighbours(graph)
		#self.build_node_to_partition_map()
		self.initialize_modality_weights(num_feature_map, feature_map)
		# self.initialize_q(seed, feature_map)
		# self.initialize_shell(feature_map)

	# def get_feature_names(self, feature_names):
	# 	# maps the feature names to node numbers
	# 	feature_name_map = {}

	# 	for i, item in enumerate(set(feature_names)):
	# 		feature_name_map[item] = -(i+1)
	# 	return feature_name_map
	def find_one_hop_neighbors(self, graph):
		# function that finds and stores the neighbours of all nodes in graph one time
		one_hop_neighbors = {}
		for node in graph.nodes:
			#print(self.graph.neighbors(node))
			one_hop_neighbors[node] = list(graph.neighbors(node))
		return one_hop_neighbors

	# def one_hop_neighbors(self, node, graph):
	# 	# function that returns the neighbours of a given node i.e evidence nodes
	# 	return graph.neighbors(node)

	def find_two_hop_neighbors(self, graph, one_hop_neighbors):
		# returns the nodes of path 2 from given node
		neighbors_path2 = {}
		for node in graph.nodes:
			nbrs = []
			for neighbor in one_hop_neighbors[node]:
				nbrs.extend([n for n in one_hop_neighbors[neighbor] if n != node and n not in self.nodes_in_q.keys()])
			neighbors_path2[node] = nbrs
		return neighbors_path2

	def num_positive_labels(self):
		# function returns the number of positive labels in the dataset
		return sum(self.labels)

	# def initialize_related_nodes(self):
	# 	# initialize the dictionary of related nodes
	# 	for data_point in range(self.num_data_points):
	# 		self.related_nodes[data_point] = [data_point]

	def initialize_q(self, graph, neighbors, one_hop_neighbors, seed, feature_map):
		# function to initialize the priority queue with seed node
		print("Initializing queue")
		evidence_score_seed_node = self.get_score(graph, seed, feature_map, one_hop_neighbors)
		self.redthread_q.put((-evidence_score_seed_node, seed))
		print("Putting node " + str(seed))
		self.update_nodes_in_q(graph, neighbors, one_hop_neighbors, feature_map, add=seed)

	def initialize_modality_weights(self, num_feature_map, feature_map):
		# function to initialize the weighs of the modalities
		print("Initializing modality weights")
		weights = np.random.random_sample(num_feature_map)
		for index, feature in enumerate(feature_map.keys()):
			self.modality_weight[feature] = weights[index]
		#print(self.feature_map.keys())

	def get_score(self, graph, node, feature_map,one_hop_neighbors):
		# function that calculates the tie-score for a given node
		#print("Getting score")
		score = 0.
		for modality in feature_map.keys():
			score += self.evidence_flow(graph, node, modality, feature_map,one_hop_neighbors) * self.get_modality_weight(modality)
		return score

	def initialize_shell(self, graph, feature_map, one_hop_neighbors):
		# function to initialize the nodes in shell (neighbours of the nodes in queue)
		print("Initializing shell")
		print("Q size = " + str(len(self.nodes_in_q)))
		print("Storing scores for each node in the Q and their neighbors")
		# print(self.nodes_in_q)
		for node, neighbor_nodes in self.nodes_in_q.items():
			# nodes_in_shell.extend(self.get_neighours(node))
			# store the score for each of the nodes in shell
			print("Q item neighborhood size = " + str(len(neighbor_nodes)))
			for nbr in tqdm(neighbor_nodes):
				self.shell[nbr] = self.get_score(graph, nbr, feature_map, one_hop_neighbors)
		print(self.shell)

			
	def build_node_to_partition_map(self):
		# mapping the evidence nodes to their partition types
		self.partition_map = {}
		for feature_type, evidence_nodes in self.feature_map.items():
			for node in evidence_nodes:
				self.partition_map[node] = feature_type

	def build_graph(self, data, make_graph, feature_names):
		# build a graph with the data points and the feature names as nodes
		# an edge exists between two nodes if one of them is an ad and the other is a feature that the ad has
		# if graph already exists in file, opens it
		if os.path.exists('models/redthread_graph.gpkl') and not make_graph:
			graph = nx.read_gpickle('models/redthread_graph.gpkl')
			neighbors = pkl.load(open('models/redthread_graph_node_neighbors.pkl','rb'))		
		else:
			num_data_points = data.shape[0]
			graph = nx.Graph() # creating an undirected graph
			#feature_nodes = [-(x+1) for x in range(len(self.feature_names))] # naming the feature nodes with negative numbers
			feature_nodes = np.array(list(map(lambda x:-(x+1), list(range(len(data[0]))))))# naming the feature nodes with negative numbers
			data_nodes = np.array(list(range(num_data_points))) # naming the ad nodes with numbers

			all_nodes = np.append(data_nodes, feature_nodes) # concatenating the two kinds of nodes
			graph.add_nodes_from(all_nodes) # adding the nodes to the graph

			nonzero_samples, nonzero_features = data.nonzero() # find non-zero features and samples and add edges between them in graph
			graph.add_edges_from(zip(data_nodes[nonzero_samples], feature_nodes[nonzero_features]))

			# for feature in tqdm(range(len(self.feature_names))):
			# 	for sample in range(self.num_data_points):
			# 		print(self.data.toarray()[sample][feature])
			# 		if self.data.toarray()[sample][feature] > 0:
			# 			self.graph.add_edge(sample, -(feature+1))
			#nx.draw(self.graph, with_labels=True)
			#plt.show()
			neighbor_evidence_nodes = self.find_one_hop_neighbors(graph)
			neighbor_ad_nodes = self.find_two_hop_neighbors(graph, neighbor_evidence_nodes)	
			#self.graph = nx.to_scipy_sparse_matrix(self.graph)
			#self.graph = csr_matrix(self.graph)
			print("Graph created and saved")
			return graph, neighbor_evidence_nodes, neighbor_ad_nodes

	def get_graph(self, data):
		# function returns the graph
		return data

	def oracle(self, query_node):
		# oracle returns the labels for the queried node
		return self.labels[query_node]

	def infer_uniformly_random(self):
		# infers the next node to be queried
		# for now, chooses a random node
		return np.random.randint(0, self.num_data_points)

	def infer_random_walk(self, seed, graph):
		# corresponds to the random walk setting
		seed_neighbors = [node for node in graph.neighbors(seed)] # find neighbours of seed node
		chosen_evidence = np.random.choice(seed_neighbors) # choose a random evidence from neighbours
		evidence_neighbors = [node for node in graph.neighbors(chosen_evidence)] # find the neighbors of the chosen evidence node
		positive_label_evidence_neighbors = [node for node in evidence_neighbors if node not in self.label_hash.keys() or self.label_hash[node] != 1] # remove those nodes with negative label from the above neighbours
		return np.random.choice(positive_label_evidence_neighbors) # randomly pick a positively labelled node from the neighborhood

	def weighted_inverse_degrees(self, node, graph):
		# this function returns 1/d_u^2 for a given evidence node
		degrees = dict(graph.degree())
		return 1./(degrees[node]*degrees[node])

	def evidence_flow(self, graph, current_node, modality, feature_map, one_hop_neighbors):
		# function that calculates the evidence support/evidence flow of a given node and a given modality
		#positive_label_nodes = [node for node, label in self.label_hash.items() if label == 1] # nodes with positive labels
		#all_nodes = [node for node in self.label_hash.keys()] # looking at all labelled nodes
		all_nodes = list(self.label_hash.keys()) # looking at all labelled nodes
		# positive_label_nodes = self.rev_label_hash[1]
		# node_neighbors = [neighbor_node for neighbor_node in self.neighbors[current_node] if neighbor_node in self.feature_map[modality]] # the neighbors of given node which belong to specific modality
		evidence_support = 0.			

		for node in all_nodes: # iterating over the nodes
			# the neighbors of positively labelled node which belong to the specified modality & current node neighbors
			common_neighbors = list(set(one_hop_neighbors[node]) & set(list(feature_map[modality])) & \
				set(one_hop_neighbors[current_node]))
			# print(common_neighbors)
			if len(common_neighbors) == 0:
				continue
			evidence_flows = []
			for n in common_neighbors:
				evidence_flows.append(self.weighted_inverse_degrees(node, graph))
			evidence_flows = np.array(evidence_flows)
			# np.array(list(map(self.weighted_inverse_degrees, common_neighbors)))
			# print(evidence_flows)
			evidence_flows *= self.label_hash[node]
			evidence_support += np.sum(evidence_flows)
			# [neighbor_node for neighbor_node in self.neighbors[node] if neighbor_node in self.feature_map[modality]]
			# finding the nodes common to both the neighborhoods
			# common_nodes = list(set(node_neighbors) & set(positive_label_node_neighbors))

			# weighted_inverse_degrees = []
			# for c_node in common_nodes: # finding the inverse degree squares for calculating the evidene flow
			# 	if self.label_hash[node] > 0:
			# 		weighted_inverse_degrees.append(1./(self.graph.degree(c_node)*self.graph.degree(c_node)))
			# 	else:
			# 		weighted_inverse_degrees.append(-1./(self.graph.degree(c_node)*self.graph.degree(c_node)))
			 
			# evidence_support += sum(weighted_inverse_degrees)
		if node not in self.evidence_flow_map.keys():
			self.evidence_flow_map[node] = {modality:evidence_support}
		else:
			self.evidence_flow_map[node][modality] = evidence_support
		return evidence_support

	def infer_weighted_random_walk(self, seed, graph):
		# corresponds to the weighted random walk setting
		seed_neighbors = [node for node in graph.neighbors(seed)] # find neighbours of seed node
		neighbor_node_weights = [float(1/graph.degree(node)) for node in seed_neighbors] # find weights of the neighbor nodes
		neighbor_node_probs = [weight/sum(neighbor_node_weights) for weight in neighbor_node_weights] # divide by the sum to make it a prob distribution
		chosen_evidence = np.random.choice(seed_neighbors, p=neighbor_node_probs) # choose a random evidence from neighbours
		evidence_neighbors = [node for node in graph.neighbors(chosen_evidence)] # find the neighbors of the chosen evidence node
		positive_label_evidence_neighbors = [node for node in evidence_neighbors if node not in self.label_hash.keys() or self.label_hash[node] != 1] # remove those nodes with negative label from the above neighbours
		return np.random.choice(positive_label_evidence_neighbors) # randomly pick a positively labelled node from the neighborhood

	def get_modality_weight(self, modality):
		# this function returns the modality weight assigned to a specific modality
		return self.modality_weight[modality]

	def update_nodes_in_q(self, graph, neighbors, one_hop_neighbors, feature_map, add=None, remove=None):
		# function that maintains all the nodes in the priority queue

		print("Updating nodes in Q")
		if remove != None:
			nbrs = neighbors[remove]
			for nbr in nbrs:
				if nbr in  self.shell.keys():
					del self.shell[nbr]
			self.nodes_in_q.pop(remove)
		elif add != None:
			if add not in self.nodes_in_q.keys():
				nbrs = neighbors[add]
				self.nodes_in_q[add] = nbrs
				for nbr in nbrs:
					self.shell[nbr] = self.get_score(graph, nbr, feature_map,one_hop_neighbors)
				# self.nodes_in_q[add] = self.get_neighours(add)
		else:
			print("Neither added nor removed node from queue")
			exit()
		return neighbors

	def update_scores_in_shell(self, graph, most_supported_modality, old_modality_weight, feature_map, one_hop_neighbors):
		# recomputing the evidence flow for all the nodes in the shell that are connected to the most supported modality
		# only these nodes need to be updated because the modality weight was updated for only one modality

		node_list = np.where(data.transpose()[most_supported_modality] > 0)[0]
		modality = list(feature_map.keys())[most_supported_modality]
		print("Updating scores in shell of size " + str(len(node_list)))
		#print(node_list)
		for node in self.shell.keys():
			#if node not in self.shell.keys():
			#	continue
			#if node in self.evidence_flow_map.keys(): # if the evidence flow for that node has already been calculated
			#	self.shell[node] -= (self.evidence_flow_map[node][modality] * old_modality_weight) # removing the old weight of the modality from the evidence score
			#	self.shell[node] += (self.evidence_flow_map[node][modality] * self.get_modality_weight(modality)) # adding the updated weight of modality
			#else:
			self.shell[node] = self.get_score(graph, node, feature_map, one_hop_neighbors)

	def update_queue(self, graph, neighbors, one_hop_neighbors, feature_map):
		# update the priority queue with the evidence flow for the given node

		# sum_evidence = 0.
		# for modality in self.feature_map.keys(): # iterate over all modalities
		# 	modality_weight = self.get_modality_weight(modality)
		# 	evidence_support = self.evidence_flow(node, modality)
		# 	sum_evidence += (evidence_support * modality_weight)
		print("Updating queue")
		print(self.label_hash)
		print(self.shell)
		for node, score in self.shell.items():
			print(self.nodes_in_q.keys())
			if node in self.label_hash.keys():
				if self.label_hash[node] == 1:
					continue
			if node in self.nodes_in_q.keys():
				continue
			
			if self.redthread_q.full():
				print("Queue is full")
				return
			# score = self.get_score(node, feature_map)
			# for score in scores:
			self.redthread_q.put((-score, node))
			print(self.redthread_q.queue)
			self.update_nodes_in_q(graph, neighbors, one_hop_neighbors, feature_map, add=node)
		

	def update_modality_weights(self, graph, picked_node, picked_node_label, feature_map, one_hop_neighbors):
		# function that updates the modality weights based on the queries label from oracle
		print("Updating modality weights")
		weighted_evidence_supports = []
		for modality in feature_map.keys():
			weighted_evidence_supports.append(self.evidence_flow(graph, picked_node, modality, feature_map,one_hop_neighbors) * self.get_modality_weight(modality))
			
		print(weighted_evidence_supports)

		most_supported_modality_index = np.argmax(weighted_evidence_supports)
		most_supported_modality = list(feature_map.keys())[most_supported_modality_index]
		old_modality_weight = self.get_modality_weight(most_supported_modality)
		# updating the weight of the most supporting modality based on the label from oracle
		if picked_node_label < 0:
			updated_modality_weight = self.learning_rate * self.get_modality_weight(most_supported_modality)
		else:
			updated_modality_weight = (2 - self.learning_rate) * self.get_modality_weight(most_supported_modality)
		self.modality_weight[most_supported_modality] = updated_modality_weight
		return most_supported_modality_index, old_modality_weight
		#print(weighted_evidence_supports)

	def update_redthread(self, graph, picked_node, picked_node_label, feature_map, neighbors, one_hop_neighbors):
		# function to update the weights of the redthread algorithm

		# UPDATING THE MODALITY WEIGHTS		
		# update the modality weight for the most supported modality
		#print("Updating modality weights")
		most_supported_modality, old_modality_weight = self.update_modality_weights(picked_node, picked_node_label, feature_map, \
			one_hop_neighbors)
		print(most_supported_modality)

		# UPDATING THE SCORES OF THE NODES IN THE SHELL
		# update the evidence flow calculation for nodes surrounding those in the queue
		#print("Updating shell scores")
		# self.update_scores_in_shell(data, most_supported_modality, old_modality_weight, feature_map)

		#print("Updating queue")
		self.update_queue(graph, neighbors, one_hop_neighbors, feature_map)

	def infer_redthread(self, graph, neighbors, one_hop_neighbors, feature_map):
		# corresponds to the redthread algorithm for inferring next node
		
		# pick the next node which has max evidence flow
		#print("Queue is empty : " + str(self.redthread_q.empty()))
		if self.redthread_q.empty():
			print("Queue is empty")
			return 
		next_node = self.redthread_q.get()[1]
		#assert next_node in self.nodes_in_q.keys()
		print("Picked node = " + str(next_node))
		self.update_nodes_in_q(graph, neighbors, one_hop_neighbors,feature_map, remove=next_node)
		return next_node


	def update_label_hash(self, node, label):
		# updates the label hash based on the inferrred nodes and oracle output
		self.label_hash[node] = label
		# self.rev_label_hash[label].append(node)

	def near_duplicate(self, curr_node):
		# function returns whether the given node is a near duplicate of any of the existing positively labelled nodes
		unigram_lengths = len(self.feature_map['desc_uni']) + len(self.feature_map['title_uni'])
		bigram_lengths = unigram_lengths + len(self.feature_map['desc_bi']) + len(self.feature_map['title_bi']) + 1
		for node, label in self.label_hash.items(): 
			if label == 1: # iterate through all positively labelled nodes
				positive_node_data = self.data.todense()[node, unigram_lengths:bigram_lengths][0]
				curr_node_data = self.data.todense()[curr_node, unigram_lengths:bigram_lengths][0]
				#print("Positive node data : " + str(positive_node_data.shape))
				#print("Current node data : " + str(curr_node_data.shape))
				inner_pdt = np.multiply(positive_node_data, curr_node_data) # dot product between the given node and the positively labelled node
				if np.count_nonzero(inner_pdt) >= int(0.95*(len(self.feature_map['desc_bi'])+len(self.feature_map['title_bi']))):
					print("non zero count = " + str(np.count_nonzero(inner_pdt)))
					return True
		#print("non zero count = " + str(inner_pdt) + " : " + str(int(0.90*(len(self.feature_map['desc_bi'])+len(self.feature_map['title_bi'])))))
		return False # if none of the nodes have positive labels or none of them are duplicates

	# def shared_evidence(self, node_i, node_j):
	# 	# calculates the shared evidence between two given nodes
	# 	sharedEvidence = []
	# 	for feature in range(self.num_features):
	# 		if node_i[feature] > 0 and node_j[feature] > 0:
	# 			sharedEvidence.append(feature)
	# 	return sharedEvidence

	# def check_related(i, j):
	# 	if i == j:
	# 		self.related_nodes[i].append(j)
	# 		return True
	# 	# look at all the related nodes from node i
	# 	nodes_related_to_node_i = self.related_nodes[i]
	# 	node_j = self.data[j]

	# 	for node in nodes_related_to_node_i:
	# 		current_node = self.data[node]
	# 		if self.oracle(self.data[j]) == 1 and len(shared_evidence(current_node, node_j)):
	# 			self.related_nodes[i].append(j)
	# 			return True
	# 	return False

