"""
Author: Yucheng Wang
Student ID: 47914111
modules.py: Implementations of Graph Convolutional Network (GCN) layers and models
This script implements a 3-layer Graph Convolutional Network (GCN) for semi-supervised 
node classification. It includes:

1. A GCNLayer class representing a single GCN layer that aggregates node features 
   from neighboring nodes using a normalized adjacency matrix.
2. A GCN class that constructs a 3-layer GCN model with batch normalization and dropout, 
   designed for node classification tasks. The final layer outputs class scores for each node, 
   which are converted to probabilities using log softmax.

The model can be used for semi-supervised learning tasks where some node labels are known 
for training, while the remaining nodes need to be classified.
"""
import torch
import torch.nn.functional as F
from torch.nn import Parameter

class GCNLayer(torch.nn.Module):
    """
    A single Graph Convolutional Network (GCN) layer with an optional activation function.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, activation=True):
        """
        Forward pass of the GCN layer.
        """
        # Multiply input features with weights
        support = torch.mm(features, self.weight)
        # Multiply adjacency matrix with transformed features
        output = torch.spmm(adj, support)
        # Apply activation if specified
        if activation:
            output = F.relu(output)
        return output


class GCN(torch.nn.Module):
    """
    A 3-layer GCN model with batch normalization and dropout
    """
    def __init__(self, in_channels, hidden_channels, num_classes, dropout_prob):
        super(GCN, self).__init__()
        # First GCN Layer
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)  # Batch normalization after first layer
        
        # Second GCN Layer
        self.conv2 = GCNLayer(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)  # Batch normalization after second layer
        
        # Third GCN Layer (output layer)
        self.conv3 = GCNLayer(hidden_channels, num_classes)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, features, adj):
        """
        Forward pass of the GCN model.
        """
        # First GCN layer with batch normalization and dropout
        x = self.conv1(features, adj)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer with batch normalization and dropout
        x = self.conv2(x, adj)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Third GCN layer (output layer without activation)
        x = self.conv3(x, adj, activation=False)
        
        return F.log_softmax(x, dim=1)  # Softmax for classification
