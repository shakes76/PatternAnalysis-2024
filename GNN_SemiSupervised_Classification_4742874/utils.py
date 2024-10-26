"""
File: utils.py
Description: Utility functions used for model implementation.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import FacebookPagePage
from torch_geometric.utils.convert import to_networkx

def display_flpp_network(flpp_dataset: FacebookPagePage) -> None:
    """
        Display the FLPP network connections via edges.
    """
    flpp_graph_nx = to_networkx(flpp_dataset[0])

    fig, ax = plt.subplots(figsize=(15, 9))
    pos = nx.spring_layout(flpp_graph_nx, iterations=15, seed=1721)

    ax.axis("off")

    plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
    nx.draw_networkx(flpp_graph_nx, pos=pos, ax=ax, **plot_options)

    plt.show()
