import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __int__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
