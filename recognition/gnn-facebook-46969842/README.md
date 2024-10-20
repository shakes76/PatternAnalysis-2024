#  Multi-layer graph neural network (GNN) for semi supervised multi-class node classification using Facebook Large Page-Page Network Data

**Table of Contents**
- [Model and Problem Description](#model-and-problem-description)
- [Model Architecture](#model-architecture)
- [Model Usage](#model-usage)
- [Results](#results)
- [References](#references)

## Model and Problem Description
### Graphs
A graph is a data structure that containts nodes and edges. Nodes can represent any arbitrary object, and edges define the relationships between two nodes, and can be either directed or undirected. They are commonly used to model problems consisting of complex relationships and interactions, such as pattern recognition, social network analysis, and more. An example of a graph is shown below,

<div style="text-align: center;">
    <img src="assets/graphexample.png" alt="Graph Example" style="width: 40%">
</div>

Due to the nature of graph data structures, they are hard to analyse. Reasons why include
  1. Graphs exist in non-euclidean space, not 2D or 3D, making visualisation and interpretation difficult
  2. Graphs are dyanmic - two very visually different graphs can have similar adjacency matrix representations
  3. Large size and dimensionality increases graph complexity for human interpretation

### Graph Neural Networks (GNNs)
Graph Neural Networks are special types of neural networks that are designed to analysis graph data structures. An input graph is passed through a series of neural networks, converted into graph embedding which allows us to maintain information on nodes, edges, etc. There are many types of GNNs, inlcuding
  1. Graph Convolutional Neural Networks (GCNs)
  2. Graph Auto-Encoder Networks
  3. Recurrent Graph Neural Networks (RGNNs)
  4. Gated Graph Neural Networks (GGNNs)

GNNs can carry out a variety of tasks, inluding
  1. Graph Classification
  2. Node Classification
  3. Link Prediction
  4. Community Detection
  5. Graph Embedding
  6. Graph Generation

<div style="text-align: center;">
    <img src="assets/GNNuses.png" alt="GNN Uses" style="width: 60%">
</div>

### The Problem
The problem is to create a suitable multi-layer graph neural network (GNN) model to carry out a semi supervised multi-class node classification using the Facebook Large Page-Page Network dataset. 

This graph is a page-page graph of verified Facebook sites, where nodes represent pages and the links are mutual likes between sites. Node features are extracted from the site descriptions that the page owners created to summarize the purpose of the site. The categories are restricted to the 4 categories defined by Facebook: 
    - Politicians
    - Governmental Organizations
    - Television Shows
    - Companies

The task related to this dataset is multi-class node classification for the 4 site categories.

## Model Architecture
The model used to solve this problem is a Graph Convolutional Neural Network (GCN). These are similar to traditional CNNs, learning features by inspecting neighboring nodes. GNNs aggregate node vectors, pass the result to the dense layer, and apply non-linearity using the activation function.

This model consists of three graph convolutional layers followed by a linear classifier. It takes as input a dataset containing node features and edge information:
- **1st Layer:** Takes input features and maps to 8 output features.
- **2nd Layer:** Takes 8 input features and maps to 8 output features.
- **3rd Layer:** Takes 8 input features and maps to 4 output features.
- **4th Layer (Classifier):** Maps 4 input features to the number of classes.

In each forward iteration, the model:
1. Extracts node features and edge information.
2. Passes through the first convolutional layer, applying the ReLU activation function.
3. Applies dropout with a probability of 0.5 to help prevent overfitting.
4. Passes through the second convolutional layer, applying ReLU and dropout again.
5. Passes through the third convolutional layer, applying ReLU to generate the final set of node embeddings.
6. Passes through to the classifier layer.

This model uses the standard Cross Entropy Loss and Adam Optimizer.
## Model Usage

1. Download the .npz data from [here](https://graphmining.ai/datasets/ptg/facebook.npz) and store it in your local directory where you have the code. This is a partially processed dataset where the features are in the form of 128 dimesion vectors
2. Train the model using
    ```python
    python train.py
    ```
    Which will saved the model to an output .pth file ```model.pth```
3. Test the model and predict, visualising using t-SNE and UMAP by running
   ```python
   python predict.py
   ```

## Results

## References
[1] A Comprehensive Introduction to Graph Neural Networks (GNNs). https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
[2] Facebook Large Page-Page Network https://snap.stanford.edu/data/facebook-large-page-page-network.html
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Training
Epoch [10/400], Loss: 1.26323, Validation Loss: 1.23775
Epoch [20/400], Loss: 1.03208, Validation Loss: 0.98408
Epoch [30/400], Loss: 0.76277, Validation Loss: 0.67652
Epoch [40/400], Loss: 0.56973, Validation Loss: 0.46284
Epoch [50/400], Loss: 0.47164, Validation Loss: 0.36143
Epoch [60/400], Loss: 0.41575, Validation Loss: 0.30780
Epoch [70/400], Loss: 0.38444, Validation Loss: 0.28174
Epoch [80/400], Loss: 0.38172, Validation Loss: 0.26687
Epoch [90/400], Loss: 0.37133, Validation Loss: 0.25546
Epoch [100/400], Loss: 0.34808, Validation Loss: 0.24646
Epoch [110/400], Loss: 0.35629, Validation Loss: 0.23923
Epoch [120/400], Loss: 0.34252, Validation Loss: 0.23486
Epoch [130/400], Loss: 0.34092, Validation Loss: 0.22914
Epoch [140/400], Loss: 0.33166, Validation Loss: 0.22596
Epoch [150/400], Loss: 0.32694, Validation Loss: 0.22643
Epoch [160/400], Loss: 0.32656, Validation Loss: 0.22386
Epoch [170/400], Loss: 0.31865, Validation Loss: 0.21791
Epoch [180/400], Loss: 0.30634, Validation Loss: 0.21633
Epoch [190/400], Loss: 0.31656, Validation Loss: 0.21400
Epoch [200/400], Loss: 0.30447, Validation Loss: 0.21124
Epoch [210/400], Loss: 0.30337, Validation Loss: 0.21200
Epoch [220/400], Loss: 0.29977, Validation Loss: 0.21104
Epoch [230/400], Loss: 0.29541, Validation Loss: 0.21014
Epoch [240/400], Loss: 0.30038, Validation Loss: 0.20703
Epoch [250/400], Loss: 0.29481, Validation Loss: 0.20667
Epoch [260/400], Loss: 0.28921, Validation Loss: 0.20640
Epoch [270/400], Loss: 0.29269, Validation Loss: 0.20486
Epoch [280/400], Loss: 0.28636, Validation Loss: 0.20538
Epoch [290/400], Loss: 0.28911, Validation Loss: 0.20671
Epoch [300/400], Loss: 0.28361, Validation Loss: 0.19912
Epoch [310/400], Loss: 0.27804, Validation Loss: 0.20120
Epoch [320/400], Loss: 0.28651, Validation Loss: 0.20144
Epoch [330/400], Loss: 0.27986, Validation Loss: 0.19939
Epoch [340/400], Loss: 0.28565, Validation Loss: 0.20001
Epoch [350/400], Loss: 0.28654, Validation Loss: 0.19993
Epoch [360/400], Loss: 0.28494, Validation Loss: 0.19804
Epoch [370/400], Loss: 0.27647, Validation Loss: 0.19675
Epoch [380/400], Loss: 0.28092, Validation Loss: 0.19440
Epoch [390/400], Loss: 0.27251, Validation Loss: 0.19472
Epoch [400/400], Loss: 0.27700, Validation Loss: 0.19314

Testing
Test Accuracy: 93.99%
