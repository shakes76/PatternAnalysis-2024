"""
A file containing the Graph Neural Network architectures to be used.
These include: GCN, GAT, SGC, JK-Nets
@author Anthony Ngo
@date 21/10/2024
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv

class GCNModel(torch.nn.Module):
    """
    A simple graph convolutional network architecture
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Apply GCN layer and ReLU
        x = F.dropout(x, training=self.training)  # Dropout for regularization
        x = self.conv2(x, edge_index)  # Second GCN layer

        return F.log_softmax(x, dim=1)  # Output class probabilities
    

class GATModelBasic(torch.nn.Module):
    """
    A simple graph attention network architecture
    Dropout -> GATConv -> Relu -> GATConv -> Softmax
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModelBasic, self).__init__()
        # 8 heads: performing multi-head-attention
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=False)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=8, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
