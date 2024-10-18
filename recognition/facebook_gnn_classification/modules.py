# modules.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4)  # First GAT layer with multi-head attention
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4)  # Second GAT layer, increasing feature capacity
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=4)  # Third GAT layer to increase depth
        self.conv4 = GATConv(hidden_dim * 4, output_dim, heads=1)  # Output layer with single head

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