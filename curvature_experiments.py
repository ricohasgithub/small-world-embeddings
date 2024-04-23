
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from smallworld import get_smallworld_graph
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def show_results(G):
    # Print the first five results
    print("Graph, first 5 edges: ")
    for n1,n2 in list(G.edges())[:5]:
        print("Ollivier-Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2]["ricciCurvature"]))

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, "ricciCurvature").values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures (Karate Club)")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights (Karate Club)")

    plt.tight_layout()

N = np.arange(100, 1000, 1)
betas = np.arange(0.025, 0.5, 0.01)

for n in N:
    for beta in betas:
        G = get_smallworld_graph(n, 10, beta)
        orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy()
        show_results(G_orc)