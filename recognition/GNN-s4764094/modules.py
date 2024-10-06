import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 64)
        self.final_class = torch.nn.Linear(64, classes)

    def forward(self, feature, edge_index):
        feature = self.conv1(feature, edge_index)
        feature = F.relu(feature)

        feature = self.conv2(feature, edge_index)
        feature = F.relu(feature)

        feature = self.conv3(feature, edge_index)
        feature = F.relu(feature)

        feature = self.conv4(feature, edge_index)
        feature = F.relu(feature)

        classification = self.final_class(feature)
        return F.log_softmax(classification, dim=1)

