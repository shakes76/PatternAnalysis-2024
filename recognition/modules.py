import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define a GCN model with four convolutional layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # Define four GCN convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First graph convolution layer + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second graph convolution layer + ReLU
        x = self.conv2(x, edge_index)
        # Apply log_softmax for classification (if required)
        return F.log_softmax(x, dim=1)