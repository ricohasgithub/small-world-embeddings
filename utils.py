
import json
import pickle
import random

import torch
import torch_geometric

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import deque
from smallworld.draw import draw_network
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

    x, y = embedding[:, 0], embedding[:, 1]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', marker='o')  # Customize color and marker here

    # Adding titles and labels
    plt.title('Circular Embedding')

    # Show the plot
    plt.grid(True)
    plt.show()

def plot_embeddings_3d(embedding):
    
    x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='red', marker='o')  # Customize color and marker here

    # Adding titles and labels
    ax.set_title('Final Embedding')

    # Show the plot
    plt.show()

g = 0

# Read graph
with open(f"dataset/graph_{g}.pickle", "rb") as file:
    data = pickle.load(file)
    # Convert to NetworkX graph
    G = nx.Graph()
    # Add edges from edge_index
    edge_index = data.edge_index.t().numpy()
    G.add_edges_from(edge_index)

    # Optional: Add node features if present
    for i, feat in enumerate(data.x):
        G.nodes[i]['feature'] = feat.numpy()

with open(f"dataset/graph_{g}_meta.json", "r") as file:
    meta = json.load(file)
    k_over_2 = meta["k_over_2"]

draw_network(G, k_over_2)
spherical_embedding = np.load(f"samples/sample_circle_{g}.npy")
final_embedding = np.load(f"samples/sample_final_{g}.npy")
plot_embeddings_2d(spherical_embedding)
plot_embeddings_3d(final_embedding)