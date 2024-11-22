# Implements GCN model based on torch
# Jiwhan Oh s4722208
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define a GCN model with four convolutional layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        '''
        Parameters
        in_channels: Feature dimension. It should match the number of features in each node in your graph
        hidden_channels: Number of channels or feature dimensions in the hidden layers of the GCN. 
        out_chnnel: Output features per node, representing the final feature dimensionality.
        '''
        super(GCN, self).__init__()
        # Define four GCN convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First graph convolution layer + ReLU

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second graph convolution layer + ReLU
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Third graph convolution layer + ReLU
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Fourth graph convolution layer
        x = self.conv4(x, edge_index)   

        # Apply log_softmax for classification (if required)
        return F.log_softmax(x, dim=1)
