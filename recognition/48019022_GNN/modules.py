"""
A file containing the Graph Neural Network architectures to be used.
These include: GCN, GAT, SGC, GraphSAGE
@author Anthony Ngo
@date 21/10/2024
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor

class GCNModel(torch.nn.Module):
    """
    A simple graph convolutional network architecture

    The model consists of:
    - Two graph convolutional layers (GCNConv):
        The first GCNConv layer projects input node features to a hidden representation, followed by ReLU activation.
        The second GCNConv layer projects the hidden representation to the output classes.
    - Dropout layer for regularization, applied between the GCN layers to prevent overfitting.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Apply GCN layer and ReLU
        x = F.dropout(x, training=self.training)  # Dropout for regularization
        x = self.conv2(x, edge_index)  # Second GCN layer

        return F.log_softmax(x, dim=1)  # Output class probabilities
    

class GATModelBasic(torch.nn.Module):
    """
    A simple graph attention network architecture

    The model consists of:
    - Two graph attention (GATConv) layers:
        The first GAT layer uses multi-head attention (8 heads) to learn weighted relations between connected nodes.
        The second GAT layer combines the attention heads and projects them to the output classes.
    - ReLU activation function applied after the first attention layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModelBasic, self).__init__()
        # 8 heads: performing multi-head-attention
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    """
    A simple Graph SAGE architecture

    The model consists of:
    - Two SAGEConv layers:
        The first layer aggregates neighborhood information and learns a hidden representation.
        The second layer aggregates hidden representations and projects them to output classes.
    - Dropout layer for regularization, applied before the first convolution.
    - ReLU activation function applied after the first layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, normalize=True)
        self.conv2 = SAGEConv(hidden_dim, output_dim, normalize=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x
    

class SGCModel(torch.nn.Module):
    """
    Implementation of Simple Graph Convolution (SGC)

    A Simple Graph Convolution (SGC) model for node classification.
    
    This model simplifies the traditional GCN by removing non-linearity between layers,
    relying on multi-hop neighborhood aggregation followed by a linear transformation.

    The model consists of:
        A K-step propagation layer (propagate_features) for neighborhood aggregation across K hops.
        A linear transformation layer that projects aggregated features to output classes.

    As the SGC is considered one of the simplest GNN architectures, this implementation avoids
    using SGConv, and attempts to build the architecture somewhat from scratch :)
    """
    def __init__(self, input_dim, output_dim, k=2):
        super(SGCModel, self).__init__()
        self.k = k
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Unpack node features and graph structure
        
        # Step 1: Feature propagation (via K-hop neighborhood aggregation)
        x = self.propagate_features(x, edge_index=edge_index)

        # Step 2: Apply linear transformation
        x = self.linear(x)
        
        return F.log_softmax(x, dim=1)  # Apply softmax for classification
    
    def propagate_features(self, x, edge_index):
        # Add self-loops to the adjacency matrix (necessary for SGC)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute the degree of each node
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle division by zero
        
        # Normalize the adjacency matrix
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Create a sparse adjacency matrix in COO format
        adj_t = SparseTensor(row=row, col=col, value=norm, sparse_sizes=(x.size(0), x.size(0)))

        # Perform feature propagation for k steps (propagating features across K hops)
        for _ in range(self.k):
            # Sparse matrix multiplication: Propagate features
            x = adj_t @ x
            
        return x

