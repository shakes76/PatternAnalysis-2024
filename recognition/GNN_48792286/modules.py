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
