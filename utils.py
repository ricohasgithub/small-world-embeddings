import torch
import torch_geometric
import geomstats

import numpy as np
import matplotlib as plt

from torch_geometric.data import Data
from geomstats.datasets.prepare_graph_data import Graph

class GS_Graph(Graph):

    def __init__(self, adjancecy, labels):
        pass

def nx_to_torch_geometric(graph, color_map):
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
    data = Data(x=node_features, edge_index=edge_index)
    return data