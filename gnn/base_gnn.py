
import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib as plt

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv_1 = GCNConv(num_node_features, 16)
        self.conv_2 = GCNConv(16, num_classes)

    def forward(self, data):
        # Retrieve data features
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), x
