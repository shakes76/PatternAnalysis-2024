# Initial GCN model using Pytorch geometric 
# Reference: Taken from https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First layer: GCN + ReLU activation
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        # Apply dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer: GCN to output logits
        x = self.conv2(x, edge_index)
        return x
