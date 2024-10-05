import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

print("GCN class loaded")

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,dropout=0.5):
        super(GCN, self).__init__()


        # Define layers
        self.convs = torch.nn.Modulelist()
        sef.convs.append(GCNConv(input_dim, hidden_dim) #first

        for _ in range(num_layers -2): #middle layers
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim)) #Output

        # Batch Normalization layers
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _in range(num_layers -1)])

        self.dropout = torch.nn.Dropout(dropout) # Add dropout with 50% prob

    def forward(self, x, edge_index):
        for i , conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index0
            x = self.bns[i](x) #apply batch normalization
            x = F.relu(x) #Activation
            x = self.dropout(x) #dropout while training

        x = self.convs[-1](x, edge_index) # Last GCN layer
        return F.log_softmax(x, dim=1)

