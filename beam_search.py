from queue import PriorityQueue
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from model import *
from torch.autograd import Variable
import sys

class Node:
    def __init__(self, parent, index, score, length):
        self.parent = parent
        self.index = index
        self.score = score
        self.length = length
    
    def get_sequence(self):
        parent = self.parent
        sequence = torch.ones(1, 1).long().fill_(self.index)

        while True:
            if parent == None:
                break
            sequence = torch.cat([torch.ones(1, 1).long().fill_(parent.index), sequence], dim=1)
            parent = parent.parent
            
            
        return sequence    



def beam_search(self, k, src, src_mask, max_len, start_symbol=0, norm=10000):

    memory = self.model.encode(src, src_mask)

    start_node = Node(None, start_symbol, 0, 0)
    
    # Create the queue used to store the nodes based on their score.
    nodes = PriorityQueue()
    # negative score here to make this a minimisation of the -log probabilities
    count=1
    nodes.put((-start_node.score, count, start_node))

    # Start here the loop over the k different options for the sequence.    
    end_nodes = PriorityQueue()

    while True:
        node = nodes.get()[2]
          
        sequence = node.get_sequence()
        out = self.model.decode(memory, src_mask, 
                            Variable(sequence).cuda() if self.parameters.cuda else Variable(sequence), 
                            Variable(subsequent_mask(sequence.size(1)).type_as(src.data)))
        prob = self.model.generator(out[:, -1])

        top_elements = torch.topk(prob, min(k, prob.size(1)))
        
        new_scores, new_words = top_elements[0], top_elements[1]
        
        for i in range(min(k, prob.size(1))):
            new_node = Node(node, new_words[0][i].item(), node.score + new_scores[0][i].item(), node.length+1)
            if new_node.length == (max_len-1):
                end_nodes.put((-new_node.score, count, new_node))
            count+=1
            nodes.put((-new_node.score, count, new_node))
            
        temp = [nodes.get() for _ in range(min(k, prob.size(1)))]
        nodes = PriorityQueue()
        for i in range(min(k, prob.size(1))):
            nodes.put(temp[i])

        if end_nodes.qsize() >= k: break
        
    Y = end_nodes.get()[2].get_sequence()

    final_y = np.around( (Y.view(1,-1).data.cpu().numpy()) / norm, decimals=5)
    return final_y

