import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
import seaborn as sns


#Graph Convolutional Network declaration 
class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCNNet, self).__init__()

        #Define convolutional layers
        self.convs = torch.nn.ModuleList()

        #Add three Layers (Maybe increased in future)
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        for conv in self.convs[:-1]:
            #ReLu activation for the first layer 
            x = F.relu(conv(x, edge_index))

            #Dropout to prevent vanishing gradient 
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, edge_index)

        #Final activation to squash values into the output range 
        return F.log_softmax(x, dim=1)


