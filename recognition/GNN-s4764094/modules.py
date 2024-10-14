import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    """
    GCN neural network model for classification tasks.

    :param classes: Number of output classes.
    :param features: Input feature size.
    """

    def __init__(self, classes, features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(features, 32)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.final_class = torch.nn.Linear(32, classes)

    def forward(self, feature, edge_index):
        """
        Forward pass through GCN model.

        :param feature: Information of input features.
        :param edge_index: Relevant indices of edges.
        :return: Log probabilities for each class.
        """
        feature = self.conv1(feature, edge_index)
        feature = F.relu(feature)

        feature = self.dropout(feature)

        classification = self.final_class(feature)
        return F.log_softmax(classification, dim=1)

    def get_embeddings(self, feature, edge_index):
        """
        Extracting node embeddings from our GCN model.

        :param feature: Information of input features.
        :param edge_index: Relevant indices of edges.
        :return: Embeddings for specified node.
        """
        feature = self.conv1(feature, edge_index)
        embeddings = F.relu(feature)
        return embeddings
