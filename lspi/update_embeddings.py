import numpy as np
import networkx as nx
import random
from sample import Sample
import math
    
    
class Embedding():
    def __init__(self, dimension, rho = 0.1, neg_samples = 5, num_nodes = 0):
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.rho = rho
        self.embeds = {}
        self.nodes = {}
        self.node_list = []
    
        for i in range(num_nodes):
            self.nodes[i] = 0
    
        self.neg_samples = 5
        for i in range(num_nodes):
            self.embeds[i] = np.random(dimension)
        
    
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))
    
    def update_single(self, source_node, target_node, label):
        
        if(self.nodes[source_node] == 0):
            self.nodes[source_node] = 1
            self.node_list.append(source_node)
    
        if(self.nodes[target_node] == 0):  
            self.nodes[target_node] = 1
            self.node_list.append(target_node)
        
        prod = np.dot(self.embeds[source_node],self.embeds[target_node])
        grad = (label - self.sigmoid(prod))*self.rho
    
        self.embeds[target_node] += grad*self.embeds[source_node]
    
    def update(self, sample):
        source_node = sample.state
        target_node = sample.next_state
        self.update_single(source_node, target_node, 1)
    
        # Negative Samples
        for i in range(self.neg_samples):
            self.update_single(source_node, random.choice(self.node_list), 0)
    
    
    def update_batch(self, samples, labels):
        for i in range(len(samples)):
            self.update(samples[i])
    
    
    
    
    
    
    
    
    
    
    