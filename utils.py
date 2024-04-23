
import random

import torch
import torch_geometric

import numpy as np
import matplotlib as plt

from collections import deque
from torch_geometric.data import Data

# Given a networkx graph with a dictionary of color mappings, return its torch_geometric Data object representation
def nx_to_torch_geometric(graph, color_map, y, num_classes=10):

    # Convert node labels to integers starting from 0
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    # Create edge index tensor
    edge_index = torch.tensor(
        [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Create node feature tensor (one-hot encoding of colors)
    # num_classes = max(color_map.values()) + 1
    node_features = torch.zeros(len(graph.nodes()), num_classes)
    for node, color in color_map.items():
        node_features[node_mapping[node], color] = 1
    
    # Create the PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, y=y)
    return data

# Compute shortcuts in the graph
def find_shortcuts(graph):
    pass

def bfs_colors(graph, num_colors):
    
    # Assign colors to nodes using a BFS approach; dictionary to store colors assigned to nodes
    colors = {node: None for node in graph.nodes()}
    nodes = list(graph.nodes())
    # Shuffle nodes to start BFS from a random node
    random.shuffle(nodes)
    
    # Queue for BFS
    queue = deque()
    for start_node in nodes:
        if colors[start_node] is None:
            colors[start_node] = 0  # Start coloring from the first color
            queue.append(start_node)
        
            while queue:
                current = queue.popleft()
                available_colors = set(range(num_colors))  # Set of all possible colors
                
                # Remove colors of adjacent nodes from available colors
                for neighbor in graph.neighbors(current):
                    if colors[neighbor] is not None and colors[neighbor] in available_colors:
                        available_colors.remove(colors[neighbor])
                
                # Try to assign a color different from its neighbors
                if not available_colors:  # If no colors are available, assign any color (this part may be improved)
                    colors[current] = random.randint(0, num_colors - 1)
                else:
                    colors[current] = min(available_colors)
                
                # Add unvisited neighbors to the queue
                for neighbor in graph.neighbors(current):
                    if colors[neighbor] is None:
                        queue.append(neighbor)
                        colors[neighbor] = (colors[current] + 1) % num_colors  # Tentatively assign a color

    return colors

def plot_data_graph(databatch):
    pass

def plot_embeddings_2d(embedding):
    pass

def plot_embeddings_3d(embedding):
    pass
