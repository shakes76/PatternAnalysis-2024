import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    '''
    Constructs a flexible GCN with customizable dimensions.
    
    Args:
        input_dim (int): Number of input features (dataset_num_features).
        hidden_dims (list of int): List of hidden layer dimensions.
        output_dim (int): Number of output classes (dataset_num_classes).
        dropout_rate (float): Dropout rate to apply during training.
    '''
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        
        torch.manual_seed(42)
        
        # Define GCN layers with the specified dimensions
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.final_conv = GCNConv(hidden_dims[2], output_dim)
        
        # Store the dropout rate
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        # First GCN layer + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second GCN layer + ReLU + Dropout
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Third GCN layer + ReLU + Dropout
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Final GCN layer (output layer)
        x = self.final_conv(x, edge_index)
        return x

# Example usage
input_dim = 128  # Number of input features
hidden_dims = [100, 64, 32]  # List of hidden layer sizes
output_dim = 4  # Number of output classes

model = GCN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=0.5)
print(model)
