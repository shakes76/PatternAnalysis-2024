"""Modules File"""
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from dataset import FacebookDataset

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for semi-supervised node classification.

    This model consists of three graph convolutional layers followed by a linear classifier. 
    It takes as input a dataset containing node features and edge information.

    Attributes:
        conv1 (GCNConv): The first graph convolutional layer.
        conv2 (GCNConv): The second graph convolutional layer.
        conv3 (GCNConv): The third graph convolutional layer.
        classifier (Linear): The final linear layer for classification.

    Parameters:
        dataset (object): The dataset that contains node features and classes.
    """
    
    def __init__(self, dataset):
        super(GCN, self).__init__()

        # Define the graph convolutional layers
        self.conv1 = GCNConv(dataset.num_features, 4) # First layer: input features to 4 output features
        self.conv2 = GCNConv(4, 4) # Second layer: 4 input features to 4 output features
        self.conv3 = GCNConv(4, 2) # Third layer: 4 input features to 2 output features
        
        # Define the final classifier layer
        self.classifier = Linear(2, dataset.num_classes)  # From 2 features to num_classes

    def forward(self, data):
        """
        Forward pass for the GCN model.

        Parameters:
            data (Data): The Data object containing node features and edge connections
        Returns:
            out (Tensor): Class scores for each node.
            h (Tensor): Final node embeddings after the last GCN layer.
        """
        x, edge = data.x, data.edge  # Extract features and edge index from the data object
        
        # Pass through the first graph convolutional layer
        h = self.conv1(x, edge)
        h = h.tanh()  # Apply activation function
        
        # Pass through the second graph convolutional layer
        h = self.conv2(h, edge)
        h = h.tanh()  # Apply activation function
        
        # Pass through the third graph convolutional layer
        h = self.conv3(h, edge)
        h = h.tanh()  # Apply activation function for final embedding
        
        # Apply the final classifier to obtain class scores
        out = self.classifier(h)

        return out, h 


if __name__ == "__main__":
    # Load the dataset
    dataset = FacebookDataset(path='facebook.npz')
    data = dataset.get_data()

    # Test making the model
    model = GCN(dataset)