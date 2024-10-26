#import packages from the library
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define Graph Neural Network model as a class
class GNN(torch.nn.Module):
    '''
     Initialize the GNN model.
        Args:
            input_dim: Number of input features for each node
            hidden_dim: Number of hidden units in the first layer
            output_dim: Number of output classes for classification
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    
    #Perform a forward pass through the network
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
