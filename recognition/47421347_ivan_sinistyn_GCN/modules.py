""" This file contains the definition of the Graph Convolutional Network
    that will be used for training and testing.
"""

import torch
import torch.nn.functional as F

# Code taken from https://github.com/gayanku/SCGC/blob/main/models.py
# Graph neural network layer
class GNNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, num_classes) -> None:
        super(GCN, self).__init__()

        self.gnn1 = GNNLayer(in_features, out_features)
        self.norm1 = torch.nn.BatchNorm1d(out_features)
        self.gnn2 = GNNLayer(out_features, out_features // 2)
        self.norm2 = torch.nn.BatchNorm1d(out_features // 2)
        self.gnn3 = GNNLayer(out_features // 2, num_classes)


    def forward(self, data):
        x, adj = data.x, data.adjacency_matrix

        # Pass the first layer
        x = self.gnn1(x, adj)
        x = self.norm1(x)
        x = torch.relu(x)
        
        # Pass the second layer
        x = self.gnn2(x, adj)
        x = self.norm2(x)
        x = torch.relu(x)

        x = self.gnn3(x, adj)
        x = torch.relu(x)

        predict = F.softmax(x, dim=1)
        return predict

