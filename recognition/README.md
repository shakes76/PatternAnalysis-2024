# Task
On Facebok Large Page-Page Network dataset, GNN (Graph Neural Netwrok) model was used to perfrom semi supervised multi-class node classification. 

# Data
The Facebook Large Page-Page Network dataset was preprocessed into the facebook.npz file by UQ, with node features reduced to 128-dimensional vectors. The data consists of 22,470 nodes and 171,002 edges, forming a directed graph with binary-labeled nodes. The graph has a density of 0.001 and a transitivity of 0.232. The dataset defines four node classes: politicians, governmental organizations, television shows, and companies. The dataset (facebook.npz) was loaded using the numpy.load() function, returning three variables: features, edges, and target, all as NumPy arrays.

Features: Shape (22,470, 128) representing the node features.
Edges: Shape (342,004, 2) representing the edge index of the directed graph.
Target: Shape (22,470,) representing the ground truth class labels for each node.

These arrays were converted into tensors for compatibility with the model. To prevent bias and overfitting, the dataset was split into training (70%), validation (20%), and test (10%) sets.

# Model
The GCN model is a type of neural network designed to work with graph-structured data. Like other neural networks, it employs convolution operations, but instead of applying them to grid-based data like images, it operates on graphs. The key distinction between grid-based models and graph-based models is that grid data relies on aggregating information from fixed-sized, regular neighborhoods, whereas graph data aggregates features from dynamically defined neighborhoods based on the graph's structure.

In a GCN, each node updates its feature representation by combining the features of its neighboring nodes. The main idea is node can improve its representation by summing up the  information from its neighbors. This aggregation process is analogous to the convolution operation in traditional CNNs, where nearby pixel values are combined to detect patterns.

The GCN algorithm can be summarized as follows:
It requires a feature matrix and an adjacency matrix as inputs. These inputs are passed through predefined convolution layers, which follow the mathematical model of the GCN. After multiple graph convolution layers, each node's feature representation is updated to reflect the information from its surrounding graph structure.


# Architecture
The GCN model is built on the neural network base class model in PytTorch.

Graph Convolution layers: performs Graph convolution which takes input and output channel as parameters. 

Relu activation funciton: activation function which outputs the input directly if positive, 0 otherwise. 

The input goes through 4 graph convolutions layers and relu functions applied after first, second and third convolution layers.
The model requests for 3 parameters which are input, hidden and output channels. Input channel value was 128 which is the feature dimension and output channels is 4 which are number of classes. 

# Hyperparameters and Functions
Optimizer: Adam optimzer 
loss: CrossEntropyLoss 
num_epochs = 500
hidden_layer = 64
learning_rate = 5e-4

# Figures and Results 
TSNE plot (pre - training)
TSNE plot (post - training)
test, valiation accuracy plot 
test, valiation loss plot 

# Libraries
torch
sklearn
matplotlib
numpy

# References
Data: https://snap.stanford.edu/data/facebook-large-page-page-network.html

Model and training inspiration: https://github.com/shakes76/PatternAnalysis-2023/pull/60/files#diff-4f1ccfdc319c84bf8878cfa900e796ff49d808dc1c85dbe0845e7d0741f2b8f6

GCN: https://arxiv.org/abs/1609.02907