# PageGCN: Semi-Supervised Node Classification for Facebook Pages

## Data background
This project uses the Facebook Large Page - Page Network dataset (https://snap.stanford.edu/data/facebook-large-page-page-network.html) to perform semi-supervised multi-class node classification using a graph convolutional network (GCN). The dataset consists of connections between Facebook pages, which are divided into the following four categories: (0) Politicians, (1) Governmental Organizations, (2) Television Shows, and (3) Companies.

## Dataset preprocessing
This section describes the preprocessing of the Facebook large-scale Page-Page network dataset for training a graph convolutional network (GCN) model. The preprocessing ensures that the input features, labels, and adjacency matrices are properly formatted and normalized to be compatible with the GCN model.

### Data loading
The dataset is stored in a compressed .npz file and loaded via numpy. The dataset contains:
**edges**: An array representing the connection relationship between nodes in the graph (i.e., edges).
**features**: The feature matrix of all nodes, each row represents the feature vector of a node.
**target**: The category label of each node, used for multi-category classification tasks.

### Data spliting
The nodes are randomly divided into training set, validation set and test set, and the division ratio is as follows:
**Training Set:** 70% of the nodes, a total of 15,728 nodes.
**Validation Set:** 15% of the nodes, a total of 3,370 nodes.
**Test Set:** 15% of the nodes, a total of 3,372 nodes.

The nodes are randomly indexed to ensure good distribution among the datasets. Masks are generated for each dataset to identify the nodes belonging to each set during training and evaluation.

### Adjacency matrix preprocessing
The graph structure is defined by the edges between nodes and represented as an adjacency matrix:
**Add Self-loops:** Add self-loops to each node to ensure that each node can consider its own characteristics during message passing.
**Normalization:** The adjacency matrix is ​​normalized using the symmetric normalization method and adjusted based on the node degree. This can prevent high-degree nodes from excessively affecting the calculation results during training and ensure stable training.
**Sparse Tensor Representation:** The normalized adjacency matrix is ​​converted to a PyTorch sparse tensor to facilitate efficient memory utilization during model training.

### Output information
Edges Shape: torch.Size([342004, 2]), indicating the connection relationship between nodes.
Feature Matrix Shape: torch.Size([22470, 128]), representing the characteristics of the node.
Labels Shape: torch.Size([22470]), represents the label of the node.
Number of Nodes: 22470
Number of Features: 128
Number of Classes: 4
Training Set Size: 15728
Validation Set Size: 3370
Test Set Size: 3372
Normalized Adjacency Matrix Shape: torch.Size([22470, 22470])
Number of Non-Zero Elements in Adjacency Matrix: 364116

