"""
Author: Ananya Dubey 
Student No : 44913083 
This script contains code to define the GCN model.  
"""
# Initial GCN model using Pytorch geometric 
# Reference: Taken from https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn 


# defining a GCN layer as a separate class 
class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        #adding batch normalisation to stabalise training
        self.bn =nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        """
        defines forward pass
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        return x 
        
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = dropout 

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        # Apply dropout for regularization
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second layer
        x = self.conv2(x, edge_index)
        return x 
    
