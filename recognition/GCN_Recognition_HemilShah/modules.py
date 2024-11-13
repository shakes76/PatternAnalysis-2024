# ====================================================
# File: modules.py
# Description: Defines various utility modules and helper functions used across the project,
#              such as custom layers, activation functions, or loss functions.
# Author: Hemil Shah
# Date Created: 14-11-2024
# Version: 1.0
# License: MIT License
# ====================================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        embeddings = F.relu(self.conv2(x, edge_index))
        out = self.conv3(embeddings, edge_index)
        return out, embeddings
