# GCN On Facebook Dataset (47801725)

## Overview 
 Graph Convolutional Networks (GCNs) are used for node classification by combining both labeled and unlabeled data within a graph to categorize nodes. These networks learn a function that maps each node's features and its topological strucutre within the graph to a corresponding label. Training involves using the known labels of a subset of nodes to refine the classification function, while also taking into account the  feature similarities with neighboring nodes. This method helps the model generalize and accurately predict the labels of nodes that lack labels, making it particularly useful when labeled data is limited.

In this report, we apply a GCN to the partially processed facebook dataset, where nodes represent official Facebook pages and edges denote mutual likes between these pages, in this specific dataset we have 4 categories and our task is to classify each node into a specfic category. 

## Description of the model 
This model is a Graph Convolutional Network (GCN) with three hidden layers and a customizable output layer for node classification on graph data. It consists of:

GCN Layers: Three GCNConv layers, each followed by ReLU activation and dropout for regularization. These layers aggregate information from neighboring nodes to capture local graph structure.
Dropout: Applied after each layer to prevent overfitting, with a configurable dropout rate.
Output Layer: A final GCN layer for classification, with output dimensions matching the number of classes.
This architecture enables flexible learning on node-based datasets by processing input features based on graph connectivity.
![Example of a GCN Model Architecture](images/The-structure-of-GCN-model-Each-node-in-the-input-graph-represents-a-sample-with-a.png)

## Data Preprocessing 
This code preprocesses graph data for training with PyTorch Geometric. It loads data from a .npz file, extracting edges, features, and labels, and converts them to PyTorch tensors. Node features are standardized using scikit-learn's StandardScaler to improve model performance. A Data object is created, storing the processed features, edge connections, and labels. Finally, RandomNodeSplit is used to generate train, validation, and test masks, splitting nodes into subsets for training and evaluation.
## Training And Validation 

## Visulization 

## Conclusion 


## Dependencies 
Python 3.8 
Pytorch: 2.0.0+cpu
scikit-learn 1.3.2 
scipy 1.11.3 
matplotlib 3.8.0 
numpy 1.24.1
## References 