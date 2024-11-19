# modules.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)  # First GAT layer with 8 attention heads
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8)  # Second GAT layer with 8 attention heads
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=8)  # Third GAT layer with 8 attention heads
        self.conv4 = GATConv(hidden_dim * 8, output_dim, heads=1)  # Output layer with single head

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index

        # First GAT Layer
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        # Second GAT Layer with Residual Connection
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = x2 + x1  # Add residual connection
        x2 = F.dropout(x2, p=0.5, training=self.training)

        # Third GAT Layer with Residual Connection
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = x3 + x2  # Add residual connection
        x3 = F.dropout(x3, p=0.5, training=self.training)

        if return_embeddings:
            return x3  # Return embeddings before the final layer

        # Output Layer
        x_out = self.conv4(x3, edge_index)
        return F.log_softmax(x_out, dim=1)