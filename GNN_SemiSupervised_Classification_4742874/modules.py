"""
File: modules.py
Description: Contains the source code for the GNN model components.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch

from torch.functional import Tensor
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj

def create_adjacency_matrix(edges) -> Tensor:
    """
        Create the adjancency matrix from the graphs edge list.

        Parameters:
            edges: List of edges to create adjacency matrix from.

        Returns:
            Adjacency matrix with self referential nodes.
    """
    # Create adjancency matrix using torch geometric function
    adjacency_matrix = to_dense_adj(edges)

    # Add identity matrix to adjacency matrix to make 
    # all nodes self referential
    adjacency_matrix += torch.eye(len(adjacency_matrix))

    return adjacency_matrix

