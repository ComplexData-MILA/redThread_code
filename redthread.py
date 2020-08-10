import numpy as np
import os
import networkx as nx
import queue as Q
import pickle as pkl
import matplotlib.pyplot as plt


class RedThread:
	def __init__(self, data, labels, seed, feature_names, feature_map, make_graph, queue_size=100, lr=0.1):
		# constructor for the class
		self.data = data
		self.labels = labels
		self.label_hash = {seed:1}
		self.num_data_points = self.data.shape[0]
		self.num_features = self.data.shape[1]
		self.related_nodes = {}
		self.modality_weight = {}
		self.feature_names = feature_names
		self.feature_map = feature_map
		self.num_feature_map = len(list(feature_map.keys()))
		self.queue_size = queue_size
		self.learning_rate = lr
		self.redthread_q = Q.PriorityQueue(maxsize=queue_size) # creating a priority queue to find the next inferred node (which has max weighted evidence flow)
		self.nodes_in_q = {}
		self.shell = {}
		#self.initialize_related_nodes()
		self.build_graph(make_graph)
		#self.build_node_to_partition_map()
		self.initialize_modality_weights()
		self.initialize_q(seed)
		self.initialize_shell()

	# def get_feature_names(self, feature_names):
	# 	# maps the feature names to node numbers
	# 	feature_name_map = {}

	# 	for i, item in enumerate(set(feature_names)):
	# 		feature_name_map[item] = -(i+1)
	# 	return feature_name_map
	def find_neighbours(self):
		# function that finds and stores the neighbours of all nodes in graph one time
		self.neighbors = {}
		for node in self.graph.nodes:
			#print(self.graph.neighbors(node))
			self.neighbors[node] = list(self.graph.neighbors(node))

	def neighbours(self, node):
		# function that returns the neighbours of a given node
		return self.neighbors[node]

	def get_neighours(self, node):
		# returns the nodes of path 2 from given node
		neighbors_path2 = []
		for neighbor in self.neighbors[node]:
			neighbors_path2.extend([n for n in self.neighbors[neighbor] if n != node and n not in self.nodes_in_q.keys()])
		return neighbors_path2

	def num_positive_labels(self):
		# function returns the number of positive labels in the dataset
		return sum(self.labels)

	# def initialize_related_nodes(self):
	# 	# initialize the dictionary of related nodes
	# 	for data_point in range(self.num_data_points):
	# 		self.related_nodes[data_point] = [data_point]

	def initialize_q(self, seed):
		# function to initialize the priority queue with seed node
		print("Initializing queue")
		evidence_score_seed_node = self.get_score(seed)
		self.redthread_q.put((-evidence_score_seed_node, seed))
		print("Putting node " + str(seed))
		self.update_nodes_in_q(add=seed)

	def initialize_modality_weights(self):
		# function to initialize the weighs of the modalities
		print("Initializing modality weights")
		weights = np.random.random_sample(self.num_feature_map)
		for index, feature in enumerate(self.feature_map.keys()):
			self.modality_weight[feature] = weights[index]
		#print(self.feature_map.keys())

	def get_score(self, node):
		# function that calculates the tie-score for a given node
		#print("Getting score")
		score = 0.
		for modality in self.feature_map.keys():
			score += self.evidence_flow(node, modality) * self.get_modality_weight(modality)
		return score

	def initialize_shell(self):
		# function to initialize the nodes in shell (neighbours of the nodes in queue)
		print("Initializing shell")
		nodes_in_shell = []
		for node, neighbor_nodes in self.nodes_in_q.items():
			# nodes_in_shell.extend(self.get_neighours(node))
			# store the score for each of the nodes in shell
			print(len(neighbor_nodes))
			for nbr in neighbor_nodes:
				self.shell[nbr] = self.get_score(nbr)
			
	def build_node_to_partition_map(self):
		# mapping the evidence nodes to their partition types
		self.partition_map = {}
		for feature_type, evidence_nodes in self.feature_map.items():
			for node in evidence_nodes:
				self.partition_map[node] = feature_type

	def build_graph(self, make_graph):
		# build a graph with the data points and the feature names as nodes
		# an edge exists between two nodes if one of them is an ad and the other is a feature that the ad has
		# if graph already exists in file, opens it
		if os.path.exists('redthread_graph.gpkl') and not make_graph:
			self.graph = nx.read_gpickle('redthread_graph.gpkl')
			self.neighbors = pkl.load(open('redthread_graph_node_neighbors.pkl','rb'))		
		else:
			self.graph = nx.Graph() # creating an undirected graph
			#feature_nodes = [-(x+1) for x in range(len(self.feature_names))] # naming the feature nodes with negative numbers
			feature_nodes = list(self.feature_names) # naming the feature nodes with negative numbers
			data_nodes = range(self.num_data_points) # naming the ad nodes with numbers
			all_nodes = np.append(data_nodes, feature_nodes) # concatenating the two kinds of nodes
			self.graph.add_nodes_from(all_nodes) # adding the nodes to the graph

			for feature in range(len(self.feature_names)):
				for sample in range(self.num_data_points):
					if self.data[sample][feature] > 0:
						self.graph.add_edge(sample, -(feature+1))
			#nx.draw(self.graph, with_labels=True)
			#plt.show()
			self.find_neighbours()
			nx.write_gpickle(self.graph, "redthread_graph.gpkl")
			pkl.dump(self.neighbors, open("redthread_graph_node_neighbors.pkl","wb"))
			print("Graph created and saved")

	def get_graph(self):
		# function returns the graph
		return self.graph

	def oracle(self, query_node):
		# oracle returns the labels for the queried node
		return self.labels[query_node]

	def infer_uniformly_random(self):
		# infers the next node to be queried
		# for now, chooses a random node
		return np.random.randint(0, self.num_data_points)

	def infer_random_walk(self, seed):
		# corresponds to the random walk setting
		seed_neighbors = [node for node in self.neighbors[seed]] # find neighbours of seed node
		chosen_evidence = np.random.choice(seed_neighbors) # choose a random evidence from neighbours
		evidence_neighbors = [node for node in self.neighbors[chosen_evidence]] # find the neighbors of the chosen evidence node
		positive_label_evidence_neighbors = [node for node in evidence_neighbors if node not in self.label_hash.keys() or self.label_hash[node] != 1] # remove those nodes with negative label from the above neighbours
		return np.random.choice(positive_label_evidence_neighbors) # randomly pick a positively labelled node from the neighborhood

	def evidence_flow(self, current_node, modality):
		# function that calculates the evidence support/evidence flow of a given node and a given modality
		#positive_label_nodes = [node for node, label in self.label_hash.items() if label == 1] # nodes with positive labels
		#all_nodes = [node for node in self.label_hash.keys()] # looking at all labelled nodes
		all_nodes = list(self.label_hash.keys()) # looking at all labelled nodes

		node_neighbors = [neighbor_node for neighbor_node in self.neighbors[current_node] if neighbor_node in self.feature_map[modality]] # the neighbors of given node which belong to specific modality
		evidence_support = 0.
		#print(list(self.neighbors[current_node]))
		#print(self.feature_map[modality].values())
		#print(node_neighbors)
		for node in all_nodes: # iterating over the nodes
			# the neighbors of positively labelled node which belong to the specified modality
			positive_label_node_neighbors = list(set(self.neighbors[node]) & set(list(self.feature_map[modality])))
			# [neighbor_node for neighbor_node in self.neighbors[node] if neighbor_node in self.feature_map[modality]]
			# finding the nodes common to both the neighborhoods
			common_nodes = list(set(node_neighbors) & set(positive_label_node_neighbors))
			#print(self.neighbors[node])
			#print(self.feature_map[modality])
			#print(common_nodes)
			weighted_inverse_degrees = []
			for c_node in common_nodes: # finding the inverse degree squares for calculating the evidene flow
				if self.label_hash[node] > 0:
					weighted_inverse_degrees.append(1./(self.graph.degree(c_node)*self.graph.degree(c_node)))
				else:
					weighted_inverse_degrees.append(-1./(self.graph.degree(c_node)*self.graph.degree(c_node)))
			 
			evidence_support += sum(weighted_inverse_degrees)

		return evidence_support

	def infer_weighted_random_walk(self, seed):
		# corresponds to the weighted random walk setting
		seed_neighbors = [node for node in self.neighbors[seed]] # find neighbours of seed node
		neighbor_node_weights = [float(1/self.graph.degree(node)) for node in seed_neighbors] # find weights of the neighbor nodes
		neighbor_node_probs = [weight/sum(neighbor_node_weights) for weight in neighbor_node_weights] # divide by the sum to make it a prob distribution
		chosen_evidence = np.random.choice(seed_neighbors, p=neighbor_node_probs) # choose a random evidence from neighbours
		evidence_neighbors = [node for node in self.neighbors[chosen_evidence]] # find the neighbors of the chosen evidence node
		positive_label_evidence_neighbors = [node for node in evidence_neighbors if node not in self.label_hash.keys() or self.label_hash[node] != 1] # remove those nodes with negative label from the above neighbours
		return np.random.choice(positive_label_evidence_neighbors) # randomly pick a positively labelled node from the neighborhood

	def get_modality_weight(self, modality):
		# this function returns the modality weight assigned to a specific modality
		return self.modality_weight[modality]

	def update_nodes_in_q(self, add=None, remove=None):
		# function that maintains all the nodes in the priority queue
		#print(add)
		#print(remove)
		print("Updating nodes in Q")
		if remove != None:
			self.nodes_in_q.pop(remove)
		elif add != None:
			if add not in self.nodes_in_q.keys():
				self.nodes_in_q[add] = self.get_neighours(add)
		else:
			print("Neither added nor removed node from queue")
			exit()

	def update_scores_in_shell(self):
		# recomputing the evidence flow for all the nodes in the shell
		print("Updating scores in shell")
		for node in self.shell.keys():
			self.shell[node] = self.get_score(node)

	def update_queue(self):
		# update the priority queue with the evidence flow for the given node
		# sum_evidence = 0.
		# for modality in self.feature_map.keys(): # iterate over all modalities
		# 	modality_weight = self.get_modality_weight(modality)
		# 	evidence_support = self.evidence_flow(node, modality)
		# 	sum_evidence += (evidence_support * modality_weight)
		print("Updating queue")
		for node, score in self.shell.items():
			if node in self.label_hash.keys() or node in self.nodes_in_q.keys():
				continue
			
			if self.redthread_q.full():
				return
			self.redthread_q.put((-score, node))
			self.update_nodes_in_q(add=node)
		

	def update_modality_weights(self, picked_node, picked_node_label):
		# function that updates the modality weights based on the queries label from oracle
		print("Updating modality weights")
		weighted_evidence_supports = []
		for modality in self.feature_map.keys():
			weighted_evidence_supports.append(self.evidence_flow(picked_node, modality) * self.get_modality_weight(modality))
			#print(self.evidence_flow(picked_node, modality))
		most_supported_modality = list(self.feature_map.keys())[np.argmax(weighted_evidence_supports)]
		# updating the weight of the most supporting modality based on the label from oracle
		if picked_node_label < 0:
			updated_modality_weight = self.learning_rate * self.get_modality_weight(most_supported_modality)
		else:
			updated_modality_weight = (2 - self.learning_rate) * self.get_modality_weight(most_supported_modality)
		self.modality_weight[most_supported_modality] = updated_modality_weight
		#print(weighted_evidence_supports)

	def update_redthread(self, picked_node, picked_node_label):
		# function to update the weights of the redthread algorithm

		# UPDATING THE MODALITY WEIGHTS		
		# update the modality weight for the most supported modality
		#print("Updating modality weights")
		self.update_modality_weights(picked_node, picked_node_label)

		# UPDATING THE SCORES OF THE NODES IN THE SHELL
		# update the evidence flow calculation for nodes surrounding those in the queue
		#print("Updating shell scores")
		self.update_scores_in_shell()

		#print("Updating queue")
		self.update_queue()

	def infer_redthread(self):
		# corresponds to the redthread algorithm for inferring next node
		
		# pick the next node which has max evidence flow
		#print("Queue is empty : " + str(self.redthread_q.empty()))
		next_node = self.redthread_q.get()[1]
		#assert next_node in self.nodes_in_q.keys()
		print("Picked node = " + str(next_node))
		self.update_nodes_in_q(remove=next_node)
		return next_node


	def update_label_hash(self, node, label):
		# updates the label hash based on the inferrred nodes and oracle output
		self.label_hash[node] = label

	def near_duplicate(self, curr_node):
		# function returns whether the given node is a near duplicate of any of the existing positively labelled nodes
		unigram_lengths = len(self.feature_map['desc_uni']) + len(self.feature_map['title_uni'])
		bigram_lengths = unigram_lengths + len(self.feature_map['desc_bi']) + len(self.feature_map['title_bi']) + 1
		for node, label in self.label_hash.items(): 
			if label == 1: # iterate through all positively labelled nodes
				positive_node_data = self.data[node][unigram_lengths:bigram_lengths]
				curr_node_data = self.data[curr_node][unigram_lengths:bigram_lengths]
				#print("Positive node data : " + str(sum(positive_node_data)))
				#print("Current node data : " + str(sum(curr_node_data)))
				inner_pdt = np.multiply(positive_node_data, curr_node_data) # dot product between the given node and the positively labelled node
				if np.count_nonzero(inner_pdt) >= int(0.90*(len(self.feature_map['desc_bi'])+len(self.feature_map['title_bi']))):
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

