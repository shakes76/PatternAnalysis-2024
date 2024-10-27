# Graph Neural Network for Classification on the Facebook Large Page-Page Network Dataset

Author: Zhe Wu
student ID: 49094642

## Project Overview
This project solves the problem of semi supervised node classification of Facebook Large Page-Page Network dataset by using GCN model. The goal is to classify nodes into multiple categories based on their features and their edges. It uses batch normalization and dropout to improve the accuracy of the model, and draws loss curves and accuracy curves, combined with embedded UMAP visualization, to help better understand the model's ability to represent data.

## Table of Contents
- [Module Architecture](#module-architecture)
- [Environment Dependencies](#environment-dependencies)
- [Model Usage](#model-usage)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [References](#references)

## Module Architecture
### Graph Neural Network（GNN）
GNN covers all neural network models for processing graph data. Its goal is to process graph data through the structure of nodes, edges, and graphs, and propagate the feature information of nodes in the graph through message passing and other methods. In addition to GCN, GNN also includes several other different models and methods, each of which has a unique way to process graph structured data, including
  1. GraphSAGE: Aggregate features by sampling neighbor nodes instead of convolving all neighbors like GCN.
  2. Graph Attention Networks (GAT): Calculate the importance of each neighbor node through the attention mechanism and weight the features of the neighbor nodes.
  3. Message Passing Neural Networks (MPNN): The message passing and aggregation process between nodes can be flexibly defined.
  4. Graph Recurrent Neural Networks (GRNN): Combine graph structure with recurrent neural network (RNN) to process time-dependent graph data.
  5. Dynamic Graph Neural Networks (DGNN): Process dynamic graph structures that change over time.
The process diagram of the GNN model performing end-to-end prediction tasks is as follows：
<div style="text-align: center;">
    <img src="images/model structure.png" alt="Epoch Losses" style="width: 60%">
</div> [2]

### Graph Convolutional Network(GCN)
GCN is a specific implementation of GNN. It is a model that processes graph-structured data based on convolution operations. The representation of each node is updated by aggregating the features of each node and its neighboring nodes. Unlike traditional convolutional neural networks that operate on regular grid data (such as images and text), GCN processes irregular graph-structured data, such as social networks and molecular structure graphs.

The core principle of GCN is to propagate and update node features layer by layer, and each layer updates the representation of the node according to the node's neighbor information. Through multi-layer graph convolution operations, the node features gradually merge the information of the surrounding nodes, thereby extracting high-order features in the entire graph structure, and finally used for tasks such as node classification, edge prediction, or overall graph classification.

### Algorithm Principle
The specific GCN structure in this project is as follows：
  1. **Convolutional layer:** There are 4 convolutional layers and each convolution layer is responsible for aggregating the features of the node and its neighboring nodes. The outputs of the 1st, 2nd, and 3rd layers are the feature representations of the hidden layer, and the output of the 4th layer is the result of node classification.
  2. **Batch Normalization:** There are 3 batch normalization layers，which perform normalization after the first three layers of convolution, helping the training process to be more stable and accelerate convergence.
  3. **Dropout:** There is a Dropout operation after each convolution layer, which randomly discards some node features during training to prevent the model from overfitting.
  4. **ReLU:** ReLU activation function is applied after each convolution layer to introduce nonlinearity, so that the model can learn more complex features.
  5. **Classification layer:** The last output layer uses a Log Softmax layer, which is responsible for outputting the category probability of the node.

The data set input contains a graph structure containing node features and edge connection information. After passing through the model, a tensor will be output, indicating the logarithmic probability of each node belonging to different categories.

### Overall framework
The model uses a four-layer GCN, combined with batch normalization and Dropout to learn node representations based on graph structure and node features. The feature vector (128 dimensions) of each node is processed by GCN, and ReLU activation and Dropout are used between layers to prevent overfitting. The dataset is randomly split into train, validation, and test sets, and a weighted loss function is used during training. The model is trained using the AdamW optimizer, and the entire process lasts for 400 epochs. The model performance is tracked by training and test loss and accuracy, and finally UMAP is used to reduce the dimension of the node embedding and visualize it.

## Environment Dependencies
The project requires the installation of the following software or packages:
- Python 3.12.4
- Pytorch 2.4.1
- Cuda 11.8 
- Numpy 1.26.4
- scikit-learn 1.5.1
- Pandas 2.2.2
- Torch Geometric 2.6.1
- UMAP-learn
- Matplotlib

## Model Usage
1. Dataset loading
   ```python
   dataset.py
   ```
   Load and preprocess graph datasets and organize the data into PyTorch Geometric Data objects, splitting them into train and test sets.

2. GCN module
   ```python
   module.py
   ```
   Graph convolutional network model with 4 layers of convolution

3. Data prediction
   ```python
   predict.py
   ```
   Load the GCN model and data, perform node classification prediction, and output the prediction results

4. Training the model
   ```python
   train.py
   ```
   Train and evaluate the GCN model, record the train and test loss and accuracy, and track the best test accuracy.

5. Visualization
   ```python
   utils.py
   ```
   Draw UMAP projections and loss and accuracy curves during training and testing

## Inputs
This project uses the Facebook Large Page-Page Network dataset provided by the course. The dataset is in the form of a 128-dimensional vector feature.The nodes represent Facebook pages, and the edges represent the likes between these pages. We need to classify them based on specific features.

The dataset was not initially divided.I split the dataset into train, validation, and test sets: **80%** for **train set**, **10%** for **validation set**, and **10%** for **test set**. This is to ensure that the nodes are reasonably allocated according to the preset ratio to maintain the balance of the data. Secondly, the dataset uses a specific labeling method to effectively prevent confusion between datasets. This can maintain randomness while making the model more universal and operable. The data layout is shown in the figure below:
<div style="text-align: center;">
    <img src="images/data example.png" alt="Epoch Losses" style="width: 60%">
</div> [2]

## Outputs
### Printing accuracy
The dataset is divided into train set, validation set and test set according to 80%, 10% and 10%. And the **learning rate** is set to **0.005**. After **400 epochs**, the best **train accuracy** is **0.9409** and the **test accuracy** is **0.9206**. The accuracy and loss values of the train set and test set are as follows:
<div style="text-align: center;">
    <img src="outputs/printing result.png" alt="Epoch Losses" style="width: 60%">
</div>

### Curve
The visualization curves of accuracy and loss value corresponding to the training set and test set of the entire training process are as follows：
<div style="text-align: center;">
    <img src="outputs/loss curve.png" alt="TSNE" style="width: 45%;">
    <img src="outputs/accuracy curve.png" alt="UMAP" style="width: 45%; margin-left: 20px;">
</div>

From the loss curve, we can see that both curves are high at the beginning, and then gradually decrease. After about 50 epochs, the **loss value** stabilizes and finally approaches **0.3**. Although the training loss is significantly lower than the test loss, this indicates that the model performs better on the training set, but there may also be some overfitting trends.

From the accuracy curve, we can see that both curves rise rapidly at the beginning, then gradually stabilize, and finally fluctuate between **0.85-0.9**. 
We visualize the output results and use UMAP to reduce the dimensionality of the high-dimensional feature vector to a two-dimensional projection:

### Umap
UMAP maps high-dimensional data to low-dimensional space by reducing dimensionality, so that the global structure and local neighborhood relationship of the data can be presented intuitively. The input result of umap is as follows:
<div style="text-align: center;">
    <img src="outputs/umap projection.png" alt="Epoch Losses" style="width: 60%">
</div>

In this graph, each cluster in the figure represents a different node category, and the color reflects the true label. Although nodes from different categories form distinguishable clusters, there are some overlapping and fuzzy areas, which indicates that the model has achieved a certain classification effect, but it may be difficult to clearly distinguish certain node categories.

## References
- [1] Distill. 'A Gentle Introduction to Graph Neural Networks', Accessed 10/27.
  https://distill.pub/2021/gnn-intro/
- [2] Boldenow, Brad. 2018. 'Simple Network Analysis of Facebook Data', Accessed 10.26.
  https://www.kaggle.com/code/boldy717/simple-network-analysis-of-facebook-data
