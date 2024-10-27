"""
File: modules.py
Description: Contains the source code for the GNN model components.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch
import torch.nn as nn

from torch.functional import Tensor

from torch_geometric.utils import to_dense_adj

def create_adjacency_matrix(edges) -> Tensor:
    """
        Create the adjancency matrix from the graphs edge list.

        Parameters:
            edges: List of edges to create adjacency matrix from.

        Returns:
            Adjacency matrix with self referential nodes.
    """
    adjacency_matrix = to_dense_adj(edges)

    # Add identity matrix to adjacency matrix to make 
    # all nodes self referential
    adjacency_matrix += torch.eye(len(adjacency_matrix))

    return adjacency_matrix

class SparseLayer(nn.Module):
    """
        Sparse Layer Module (Used to construct GNN layers)
    """
    def __init__(self, dim_in: int, dim_out: int) -> None:
        """
            Ini
            Parameters:
                dim_in: the size of the vector input into the GNN.
                dim_out: the output catergorisation vector size.

        """
        super(SparseLayer, self).__init__()

        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x: Tensor, adjacency_matrix: Tensor) -> Tensor:
        """
            Forwared training pass of the GNN Sparse Layer

            Parameters:
                x: The training tensor.
                adjacency_matrix: The graph adjacency matrix.

            Returns:
                The tensor after the sparse layer has been applied to the input.
        """
        x = self.linear(x)

        # Sparse matrix multiply the linearised 
        # input with the adjacency matrix
        x = torch.sparse.mm(adjacency_matrix, x)

        return x

class GNN(nn.Module):
    """
        Graphical Neural Network Model
    """
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int) -> None:
        """
            Initialise the GNN and create sparse layers for training.

            Parameters:
                dim_in: The size of the vector input into the gnn.
                dim_hidden: The size of the hidden layer vector to transform the gnn to.
                dim_out: The output catergorisation vector size.
        """
        super(GNN, self).__init__()

        self.gnn_1 = SparseLayer(dim_in, dim_hidden)
        self.gnn_2 = SparseLayer(dim_hidden, dim_out)

    def forward(self, x: Tensor, adjacency_matrix: Tensor) -> Tensor:
        """
            Forward training pass of the GNN model.

            Parameters:
                x: The training tensor.
                adjacency_matrix: The graph adjacency matrix.

            Returns:
                The tensor after the sparse layers have been applied to the input.
        """

        h = self.gnn_1(x, adjacency_matrix)
        h = torch.relu(h)
        h = self.gnn_2(h, adjacency_matrix)

        return nn.functional.log_softmax(h, dim=1)
