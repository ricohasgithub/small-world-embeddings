
import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib as plt

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool

# Partial implementation of the Knot Embedding GNN
class KnotGCN(nn.Module):
    
    def __init__(self, num_classes, num_node_features):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 2)
        
        self.conv3 = GCNConv(2, 3)
        self.linear = nn.Linear(3, num_classes)

    def forward(self, data):
        # Retrieve data features
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.normalize(x)
        embed_1 = x.clone().detach()

        x = self.conv3(x, edge_index)

        pooled = global_mean_pool(x, batch=None)
        logits = self.linear(pooled)

        return F.log_softmax(logits, dim=1), (x, embed_1)
