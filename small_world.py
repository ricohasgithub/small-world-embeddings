
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from smallworld.draw import draw_network
from smallworld import get_smallworld_graph

# Define network parameters
N = 21
k_over_2 = 2
betas = [0, 0.025, 1.0]
labels = [r"$\beta=0$", r"$\beta=0.025$", r"$\beta=1$"]
focal_node = 0

small_world = get_smallworld_graph(N, k_over_2, 0.025)
adjacency = nx.adjacency_matrix(small_world).todense()
degree = np.diag([small_world.degree(n) for n in small_world.nodes()])
laplacian = nx.laplacian_matrix(small_world).todense()

# Use a greedy coloring algorithm to color the nodes
coloring = nx.coloring.greedy_color(small_world, strategy="saturation_largest_first")

# Map the coloring to two colors (e.g., red and blue)
color_map = {node: "red" if color == 0 else "blue" for node, color in coloring.items()}

# Draw the graph with the colored nodes
nx.draw(small_world, with_labels=True, node_color=list(color_map.values()))
plt.show()



fig, ax = plt.subplots(1,3,figsize=(9,3))

# Iterate over graph parameter set
for ib, beta in enumerate(betas):

    # Generate small-world graphs and draw
    G = get_smallworld_graph(N, k_over_2, beta)
    draw_network(G, k_over_2, focal_node=focal_node, ax=ax[ib])

    ax[ib].set_title(labels[ib],fontsize=11)

# Plot generated graphs
plt.subplots_adjust(wspace=0.3)
plt.show()
