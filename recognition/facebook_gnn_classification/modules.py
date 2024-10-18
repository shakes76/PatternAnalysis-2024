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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)