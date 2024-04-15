
import json
import pickle

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
    def __init__(self, graphs, max_n, root="./datasets", transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.max_n = max_n
        if graphs is not None:
            self.data_list = graphs
        else:
            # Load dataset from root directory
            pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def get_sw_graph(n, k_over_2, beta, y, coloring_strategy="saturation_largest_first"):
    small_world = get_smallworld_graph(n, k_over_2, beta)
    coloring = nx.coloring.greedy_color(small_world, strategy=coloring_strategy)
    return nx_to_torch_geometric(small_world, coloring, y), small_world, coloring

def generate_sw_dataset(filepath, num_graphs, max_n, beta_threshold,
                        n_mean, n_variance,
                        k_over_2_mean, k_over_2_variance,
                        beta_mean, beta_variance,
                        coloring_strategy="saturation_largest_first"):

    graphs = []
    for i in range(num_graphs):

        # Sample n, k_over_2, beta
        sample_n = max(10, min(int(round(np.random.normal(n_mean, np.sqrt(n_variance)))), max_n))
        sample_k_over_2 = max(1, int(round(np.random.normal(k_over_2_mean, np.sqrt(k_over_2_variance)))))
        sample_beta = max(0.025, min(np.random.normal(beta_mean, np.sqrt(beta_variance)), 0.4))

        # Get ground truth - set threshold on beta
        y = 1 if sample_beta >= beta_threshold else 0
        graph, _, coloring = get_sw_graph(sample_n, sample_k_over_2, sample_beta, y, coloring_strategy=coloring_strategy)

        # Save graph to filepath
        pickle.dump(graph, open(f"{filepath}/graph_{i}.pickle", "wb"))
        # Writing JSON data
        with open(f"{filepath}/graph_{i}.json", "w") as coloring_file:
            json.dump(coloring, coloring_file)
        graphs.append(graph)
        print(f"Graph: {i} generated, n={sample_n}, k_over_2={sample_k_over_2}, beta={sample_beta}")
    
    return SmallWorldDataset(graphs, max_n)

if __name__ == "__main__":
    generate_sw_dataset("./dataset", 100, 50, 0.25,
                        25, 10,
                        5, 2,
                        0.25, 0.1)
