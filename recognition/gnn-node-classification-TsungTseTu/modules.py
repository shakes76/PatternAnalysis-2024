import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

print("GCN class loaded")

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5) # Add dropout with 50% prob

    def forward(self, x, edge_index):
        # First GCN layer+ RELU active
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x) #apply dropout after relu

        # Second GCN layer + log_softmax for multiclassification
        x = self.conv2(x, edge_index)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Fourth GCN layer + log_softmax
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

