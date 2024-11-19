# -----------------------------------------------------------
# Project: Mixed Graph Neural Networks for Node Classification
# Filename: modules.py
# Author: Tsung-Tse Tu
# Student ID: s4780187
# Date: October 2024 (Last edited: 10/21/2024)
# Description: This file contains the MixedGNN class which combines 
#              GCN, GAT, and GraphSAGE layers for multi-layered graph 
#              neural network-based node classification
# -----------------------------------------------------------


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

print("GCN class loaded")

class MixedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, num_gat_layers, num_sage_layers, heads, dropout):
        super(MixedGNN, self).__init__()

        # GCN layers
        self.gcn_layers = torch.nn.ModuleList()
        for i in range(num_gcn_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # GAT layers
        self.gat_layers = torch.nn.ModuleList()
        for i in range(num_gat_layers):
            if i == 0 and num_gcn_layers == 0:
                # If no GCN layers, first GAT layer takes the input features directly
                self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            else:
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))


        # GraphSAGE layers
        self.sage_layers = torch.nn.ModuleList()
        for i in range(num_sage_layers):
            if i == 0 and num_gcn_layers == 0 and num_gat_layers == 0:
                self.sage_layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                # Ensure GraphSAGE receives the correct input dimensions
                if num_gat_layers > 0:
                    self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))
                else:
                    self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Apply GCN layers
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply GAT layers
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply GraphSAGE layers
        for layer in self.sage_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final output layer
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)
