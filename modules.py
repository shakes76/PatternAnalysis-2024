import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        # GAT layer
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)
        # GAT final layer
        self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1 attention layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU activation
        # 2 attention layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
