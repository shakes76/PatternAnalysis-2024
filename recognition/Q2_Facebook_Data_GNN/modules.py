import torch 
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, GATConv
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.utils import dropout_adj, dropout_edge


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

    def __init__(self, input_dim, hidden_dim, output_dim= 4, dropout=0.65, edge_dropout=0.4):
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
        output_dim: int 
            The dimensionality of the output (number of classes or regression output)
        num_layers : int, optional (default=6)
            The number of graph convolutional layers in the network.
        dropout : float, optional (default=0.65)
            The dropout probability used during training to prevent overfitting.
        edge_dropout : float, optional (default=0.4)
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

        #output layer
        self.convs.append(GCNConv(hidden_dim[-1], output_dim)) 
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the Enhanced GNN model.
        This function performs the forward pass of the Enhanced GNN model by applying the
        graph convolution layers to the input features.
        Different action functions are used to test the best model for the given dataset.
        Option 1: ReLU
        Option 2: ELU
        Option 3: Swish
        Option 4: Leaky ReLU
        The activation function with the best performance is used in the final model.
        Other activation functions are commented out in the code.
        Only Leaky ReLU is used in the final model

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
        # residual = x 
        # adding residual is not used
        for i, conv in enumerate(self.convs[:-1]):
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
            x = conv(x, edge_index)

            # using relu - option 1 - not used
            # x = F.relu(x)

            # USING elu - option 2 - not used
            # x = F.elu(x)

            # using swish - option 3 - not used
            # x = x * F.sigmoid(x)

            # using leaky relu
            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # adding residual connection to avooid oversmmoothing - not used
            # did not improve the model performance
            # x = x + residual if i>0 else x
            # residual = x

        # O/P layer
        x = self.convs[-1](x, edge_index)
        return x

class AdvanceGATModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim= 4, heads = 8, dropout=0.65):
        """
        This function construct the adcanced structure of
        the Graph Attention Network (GAT) by stacking GAT convolution layers
        with given dropout rate.

        Parameters:
        -----------
        input_dim : int
            The dimensionality of the input node features
        hidden_dim : int
            The dimensionality of the hidden layer(s)
        output_dim: int (default=4)
            The dimensionality of the output (number of classes or regression output)
        heads : int, optional (default=8)
            The number of attention heads in the GAT layer.
        dropout : float, optional (default=0.65)
            The dropout probability used during training to prevent overfitting.
        Returns:
        --------
        None

        """
        super(AdvanceGATModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim[0], heads = heads, dropout = dropout))  

        # hiddn layers with varying hidden dimensions
        for i in range(1, len(hidden_dim)):
            self.convs.append(GATConv(hidden_dim[i-1] * heads, hidden_dim[i], heads = heads, dropout = dropout))

        #output layer
        self.convs.append(GATConv(hidden_dim[-1] * heads, output_dim, heads = heads, dropout = dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.
        This function performs the forward pass of the GAT model by applying the
        graph attention layers to the input features.
        Different action functions are used to test the best model for the given dataset.
        Option 1: ReLU
        Option 2: ELU
        Option 3: Swish
        Option 4: Leaky ReLU
        The activation function with the best performance is used in the final model.
        Other activation functions are commented out in the code.
        Only Leaky ReLU is used in the final model

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
        # adding residual is not used
        # residual = x
        for i, conv in enumerate(self.convs[:-1]):
            edge_index, _ = dropout_edge(edge_index, training=self.training)
            x = conv(x, edge_index)

            # using relu - option 1 - not used
            # x = F.relu(x)

            # using elu - option 2 - not used
            # x = F.elu(x)

            # using swish - option 3 - not used
            # x = x * F.sigmoid(x)

            # using leaky relu
            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # adding residual connection to avooid oversmmoothing
            # did not improve the model performance - not in used
            # x = x + residual if i>0 else x
            # residual = x

        # O/P layer
        x = self.convs[-1](x, edge_index)
        return x


if __name__ == '__main__':
    """
    This block of code is used to call the implementation of the GNNModel class.
    Three different models are implemented in this block to test the best model for the given dataset.
    1. Basic GNN model - not in use
    2. Advanced GNN model - model used in the final implementation
    3. Advanced GAT model - not in use

    The best model is selected based on the performance of the model on the given dataset.
    Other models are commented out in the code.
    """
    # Basic GNN - not used
    # model = GNNModel(input_dim=128, hidden_dim=64, output_dim=4, num_layers=3)

    # Advanced GNN - not used
    model = AdvanceGNNModel(input_dim=128, hidden_dim=[512])

    # Advanced GAT
    # model = AdvanceGATModel(input_dim=128, hidden_dim=[128,128])

    print(model)