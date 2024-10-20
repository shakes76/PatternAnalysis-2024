# Aim
The aim of this project is to create a suitable multi-layer graph neural network (GNN) model to carry out a semi supervised multi-class node classification using Facebook Large Page-Page Network dataset with reasonable accuracy. After a model that can achieve sufficient accuracy is trained, a t-Distributed Stochastic Neighbor Embedding (t-SNE) should be created in order to effectively visualize the results. Should the model meet the requirements, this visualization should display different classes grouped in distinct clusterings to demonstrate how well the model captures features in its latent space.

The Facebook Large Page-Page Network dataset provided by Stanford University is a graph consisting of nodes and edges. There are no edge weightings and each node contains a label from 0 - 3 describing 4 different site catagories being politicians, governmental organizations, television shows and companies. When importing graph data, it is important to note than storing the graph as an adjacency matrix can be extremely costly as it is mostly sparse. Thus, this implementation requires a partially processed dataset where the features are in the form of 128 dim vectors. The exact relationship between nodes is described by dataset compiler Luka Leskovec as 'Nodes represent official Facebook pages while the links are mutual likes between sites. Node features are extracted from the site descriptions that the page owners created to summarize the purpose of the site'. It is expected that mutual likes between pages correspond to similar page catagories, hence given unlabeled nodes, a trained GNN should be able to accurately predict what category a page belongs to given its mutual likes. 

The specific GNN chosen for classifying this dataset is a Graph Convolutional Network (GCN). The advantage of a GCN (Particularly for a large dataset like this) is that it it allows the traditional application of a grid-like convolution on an irregular graph like structure in order to extract and learn deep features in the data. The method uses a sliding kernel similar to a CNN, however instead of aggregating pixel information it aggregates neighbor information, where each node updates its representation by aggregating the features of its neighboring nodes, thus learning a nodes embedding to capture both the node's features and its local graph structure.
# Files

## Requirements
Requirements to run this project can be found in the requirements.txt file.
## dataset.py

## modules.py

## train.py

## predict.py

# Results 

# Discussion of Results 