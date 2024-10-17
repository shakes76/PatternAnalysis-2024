import torch 
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class GNNModel(torch.nn.Module): 
    """
    This class defines the basic Graph Convolutional Network (GCN) model.
    The model consists of a stack of graph convolutional layers with ReLU activation.
    Dropout is applied after each layer to prevent overfitting.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5): 
        """
        Initilise the basic GNN model with a stack of graph convolution layers.

        This function construct the basic structure of 
        the Graph Convolutional Network (GCN) by stacking GCN convolution layers 
        with given dropout rate.
        
        Parameters:
        -----------
        input_dim : int
            The dimensionality of the input node features
        hidden_dim : int
            The dimensionality of the hidden layer(s)
        output_dim, 
            The dimensionality of the output (number of classes or regression output)
        num_layers : int, optional (default=2)
            The number of graph convolutional layers in the network.
        dropout : float, optional (default=0.5)
            The dropout probability used during training to prevent overfitting.

        Returns:
        --------
        None

        """
        super(GNNModel, self).__init__() 
        self.convs = torch.nn.ModuleList() # A list to store the GCNConv layers

        # Input layer (from input_dim to hidden_dim)
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2): 
            self.convs.append(GCNConv(hidden_dim, hidden_dim)) # Hidden layers
        self.convs.append(GCNConv(hidden_dim, output_dim)) # O/P layer            
        self.dropout = dropout # Dropout probability
            
    def forward(self, x, edge_index):
        """
        Forward pass of the GNN model.
        This function performs the forward pass of the GNN model by applying the
        graph convolution layers to the input features.

        Parameters:
        ----------- 
        x : torch.Tensor
            The input node features
        edge_index : torch.Tensor
            The edge index tensor

        Returns:    
        --------
        torch.Tensor
            The output tensor of the model

        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # O/P layer
        x = self.convs[-1](x, edge_index)
        return x
    

class AdvanceGNNModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim= 4, num_layers=6, dropout=0.7):
        """
        Initilise the advance GNN model with a stack of graph convolution layers.

        This function construct the adcanced structure of 
        the Graph Convolutional Network (GCN) by stacking GCN convolution layers 
        with given dropout rate.
        
        Parameters:
        -----------
        input_dim : int
            The dimensionality of the input node features
        hidden_dim : int
            The dimensionality of the hidden layer(s)
        output_dim, 
            The dimensionality of the output (number of classes or regression output)
        num_layers : int, optional (default=6)
            The number of graph convolutional layers in the network.
        dropout : float, optional (default=0.7)
            The dropout probability used during training to prevent overfitting.
        Returns:
        --------
        None
        """
        super(AdvanceGNNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim[0]))

        ## hiddn layers with varying hidden dimensions
        for i in range(1, len(hidden_dim)):
            self.convs.append(GCNConv(hidden_dim[i-1], hidden_dim[i]))
        self.convs.append(GCNConv(hidden_dim[-1], output_dim)) # O/P layer
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the Enhanced GNN model.
        This function performs the forward pass of the Enhanced GNN model by applying the
        graph convolution layers to the input features.

        Parameters:
        ----------- 
        x : torch.Tensor
            The input node features
        edge_index : torch.Tensor
            The edge index tensor

        Returns:    
        --------
        torch.Tensor
            The output tensor of the model

        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # O/P layer
        x = self.convs[-1](x, edge_index)
        return x
    



if __name__ == '__main__':
    """
    This block of code is used to call the implementation of the GNNModel class.
    """
    # model = GNNModel(input_dim=128, hidden_dim=64, output_dim=4, num_layers=3)
    model = AdvanceGNNModel(input_dim=128, hidden_dim=[128,256,512,128,64])
    print(model)