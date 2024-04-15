
import torch
import torch_geometric

import numpy as np
import networkx as nx
import matplotlib as plt

from torch_geometric.data import Data, Dataset, DataLoader
from smallworld.draw import draw_network
from smallworld import get_smallworld_graph

from utils import nx_to_torch_geometric

class SmallWorldDataset(Dataset):
    def __init__(self, graphs, max_n, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = graphs
        self.max_n = max_n

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

def get_sw_graph(n, k_over_2, beta, y, coloring_strategy="saturation_largest_first"):
    small_world = get_smallworld_graph(n, k_over_2, beta)
    coloring = nx.coloring.greedy_color(small_world, strategy=coloring_strategy)
    return nx_to_torch_geometric(small_world, coloring, y)

def generate_sw_dataset(num_graphs, max_n, beta_threshold,
                        n_mean, n_variance,
                        k_over_2_mean, k_over_2_variance,
                        beta_mean, beta_variance,
                        coloring_strategy="saturation_largest_first"):

    graphs = []
    for _ in range(num_graphs):

        # Sample n, k_over_2, beta
        sample_n = min(int(round(np.random.normal(n_mean, np.sqrt(n_variance)))), max_n)
        sample_k_over_2 = int(round(np.random.normal(k_over_2_mean, np.sqrt(k_over_2_variance))))
        sample_beta = np.random.normal(beta_mean, np.sqrt(beta_variance))

        # Get ground truth - set threshold on beta
        y = 1 if sample_beta >= beta_threshold else 0
        graphs.append(get_sw_graph(sample_n, sample_k_over_2, sample_beta, y, coloring_strategy=coloring_strategy))
    
    return SmallWorldDataset(graphs, max_n)
