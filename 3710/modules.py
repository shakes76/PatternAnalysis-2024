import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(GNN, self).__init__()
        
        # Define the first graph convolutional layer and batch normalization
        self.conv1 = GCNConv(in_channels, 64)
        self.batch_norm1 = BatchNorm(64)
        
        # Define the second graph convolutional layer and batch normalization
        self.conv2 = GCNConv(64, 32)
        self.batch_norm2 = BatchNorm(32)
        
        # Define the third graph convolutional layer and batch normalization
        self.conv3 = GCNConv(32, 16)
        self.batch_norm3 = BatchNorm(16)
        
        # Define the fourth (output) graph convolutional layer
        self.conv4 = GCNConv(16, out_channels)
        
        # Store the dropout rate
        self.dropout_rate = dropout_rate

    def forward(self, data):
        # Extract node features and edge indices from the data object
        x, edge_index = data.x, data.edge_index
        
        # First layer: graph convolution + batch normalization + ReLU activation + dropout
        x = self.conv1(x, edge_index)  # Apply the first GCN layer
        x = self.batch_norm1(x)        # Apply batch normalization
        x = F.relu(x)                  # Apply ReLU activation function
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Apply dropout
        
        # Second layer: graph convolution + batch normalization + ReLU activation + dropout
        x = self.conv2(x, edge_index)  # Apply the second GCN layer
        x = self.batch_norm2(x)        # Apply batch normalization
        x = F.relu(x)                  # Apply ReLU activation function
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Apply dropout
        
        # Third layer: graph convolution + batch normalization + ReLU activation + dropout
        x = self.conv3(x, edge_index)  # Apply the third GCN layer
        x = self.batch_norm3(x)        # Apply batch normalization
        x = F.relu(x)                  # Apply ReLU activation function
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Apply dropout
        
        # Output layer: graph convolution + log_softmax activation
        x = self.conv4(x, edge_index)  # Apply the fourth (output) GCN layer
        return F.log_softmax(x, dim=1)  # Apply log softmax for output probabilities


