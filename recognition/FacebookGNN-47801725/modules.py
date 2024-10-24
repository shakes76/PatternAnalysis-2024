# modules.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class EnhancedGCN(torch.nn.Module):
    """
    Enhanced GCN model with multiple GCN layers, batch normalization, and dropout.

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list of int): List containing the number of hidden units for each layer.
        output_dim (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(EnhancedGCN, self).__init__()
        torch.manual_seed(123)

        # Ensure that the number of layers in hidden_dims is consistent
        assert len(hidden_dims) == 4, "hidden_dims should have exactly 4 elements"

        # Define GCN layers and batch normalization
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.bn1 = BatchNorm(hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.bn2 = BatchNorm(hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.bn3 = BatchNorm(hidden_dims[2])
        self.conv4 = GCNConv(hidden_dims[2], hidden_dims[3])
        self.bn4 = BatchNorm(hidden_dims[3])
        self.conv5 = GCNConv(hidden_dims[3], output_dim)

        # Store the dropout rate for later use
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        """
        Forward pass through the enhanced GCN model.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge index tensor defining the graph connectivity.

        Returns:
            torch.Tensor: Log probabilities for each class.
        """
        # First GCN layer + BatchNorm + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Second GCN layer + BatchNorm + ReLU + Dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Third GCN layer + BatchNorm + ReLU + Dropout
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Fourth GCN layer + BatchNorm + ReLU + Dropout
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Output layer (no activation, as we use log_softmax)
        x = self.conv5(x, edge_index)
        return F.log_softmax(x, dim=1)
