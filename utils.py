import torch
import torch_geometric
import geomstats

import numpy as np
import matplotlib as plt

from torch_geometric.data import Data
from geomstats.datasets.prepare_graph_data import Graph

# Given a networkx graph with a dictionary of color mappings, return its torch_geometric Data object representation
def nx_to_torch_geometric(graph, color_map, y):

    # Convert node labels to integers starting from 0
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    # Create edge index tensor
    edge_index = torch.tensor(
        [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Create node feature tensor (one-hot encoding of colors)
    num_classes = max(color_map.values()) + 1
    node_features = torch.zeros(len(graph.nodes()), num_classes)
    for node, color in color_map.items():
        node_features[node_mapping[node], color] = 1
    
    # Create the PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, y=y)
    return data

# Compute shortcuts in the graph
def find_shortcuts(graph):
    pass
