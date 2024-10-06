import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, classes, features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(features, 64)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv2 = GCNConv(64, 32)
        self.final_class = torch.nn.Linear(32, classes)

    def forward(self, feature, edge_index):
        feature = self.conv1(feature, edge_index)
        feature = F.relu(feature)

        feature = self.dropout(feature)

        feature = self.conv2(feature, edge_index)
        feature = F.relu(feature)

        classification = self.final_class(feature)
        return F.log_softmax(classification, dim=1)

