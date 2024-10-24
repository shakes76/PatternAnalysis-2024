import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5, num_layers=3):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
          self.convs.append(GCNConv(hidden_channels, hidden_channels))
          self.convs.append(GCNConv(hidden_channels, out_channels)
          self.bns = torch.nn.ModuleList([BatchNorm(hidden_channels) for _ in range(num_layers - 1)])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = self.bns[0](x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        for i in range(1, self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
