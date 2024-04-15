
import torch
import torch_geometric

import numpy as np
import networkx as nx
import matplotlib as plt

from torch_geometric.data import Data, Dataset
from smallworld.draw import draw_network
from smallworld import get_smallworld_graph

from utils import nx_to_torch_geometric

from torch_geometric.data import Dataset, Data

class SmallWorldDataset(Dataset):
    def __init__(self, graphs, max_n):
        super(SmallWorldDataset, self).__init__()
        self.data_list = graphs
        self.max_n = max_n

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

def get_sw_graph(n, k_over_2, beta):
    small_world = get_smallworld_graph(n, k_over_2, beta, coloring_strategy="saturation_largest_first")
    coloring = nx.coloring.greedy_color(small_world, strategy="saturation_largest_first")
    return nx_to_torch_geometric(small_world, coloring)

def generate_sw_dataset(num_graphs, max_n,
                        n_mean, n_variance,
                        k_over_2_mean, k_over_2_variance,
                        beta_mean, beta_variance,
                        coloring_strategy="saturation_largest_first"):

    graphs = []
    for _ in range(num_graphs):
        # Sample n, k_over_2, beta
        sample_n = min(int(round(np.random.normal(n_mean, np.sqrt(n_variance)))), max_n)
        sample_k_over_2 = int(round(np.random.normal(k_over_2_mean, np.sqrt(k_over_2_variance))))
        sample_beta = int(round(np.random.normal(beta_mean, np.sqrt(beta_variance))))
        graphs.append(get_sw_graph(sample_n, sample_k_over_2, sample_beta, coloring_strategy=coloring_strategy))
    
    return SmallWorldDataset(graphs, max_n)
