'''
Author: Kangqi Wang
Student Number: 48300588

This script is about building a Graph Convolutional Network (GCN) 
model using PyTorch and PyTorch Geometric.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree

class CustomGCNConv(nn.Module):
    """
    Custom GCN Convolution Layer implementing the propagation rule:
    H^{(l+1)} = D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)}
    """

    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Weight parameter
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Compute the convolution
        x = torch.mm(x, self.weight)
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.view(-1, 1))
        return out

class GNNModel(nn.Module):
    """
    GNN Model using the CustomGCNConv layer, incorporating ideas from the reference codes.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GNNModel, self).__init__()
        self.conv1 = CustomGCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = CustomGCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = CustomGCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First convolutional layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second convolutional layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third convolutional layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final output layer
        x = self.fc(x)
        return x

    def accuracy(self, y_true, y_pred):
        return (y_true == y_pred).sum().item() / len(y_true)

