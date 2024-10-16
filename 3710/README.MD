# Multi-Class Node Classification of Facebook Network Dataset Using GCN

## Dataset information
This project uses the Facebook Large Page Network, which is a graph of pages on verified Facebook websites. The nodes represent official Facebook pages and the edges between nodes represent mutual likes between websites. The dataset is divided into 4 different categories: politicians, government organizations, TV shows and companies. The dataset consists of 22,470 nodes and 171,002 edges, each node has 128 features extracted from the website description created by the page owner.



## Algorithm

This project uses Graph Convolutional Network (GCN) for node classification, which is an algorithm that captures the dependencies between nodes through their connectivity. The model performs semi-supervised multi-class node classification on the Facebook dataset with the aim of categorizing each node into one of the four previously mentioned classes. Finally TSNE is used to visualize the data from high dimensional to two dimensional.


## Problem That It Solves

The problem solved in this project is the multi-category node classification problem, where each Facebook page is categorized into one of four categories.GCN solves the semi-supervised node classification problem using a small fraction of labeled nodes. The model is useful in a wide range of application scenarios, including social network analysis, advertisement targeting, and other tasks that require classification in complex graph structures.

## How it Works

GCN works by first initializing its feature representation for each node. Through multiple graph convolutional layers, the model continuously updates the representation of each node and its neighboring nodes by aggregating the feature information of these nodes layer by layer. Each layer of convolution operation learns the local structure information of the graph by aggregating the neighboring features. A total of three hidden layers and one output layer are used in this project. Each layer is responsible for extracting different levels of feature information, allowing the model to progressively capture complex inter-node relationships.

After feature aggregation, the model uses ReLU activation function for introducing nonlinearity, Dropout for preventing overfitting, and batch normalization to optimize the training process. And the probability of each category is assigned to the nodes in the last layer of the GCN.

During the training process, the model learns the optimal parameters by minimizing the cross-entropy loss, and also uses forward propagation and the Adam optimizer, among others, to update the parameters step by step.

## Data Pre-Processing

The data is loaded from an .npz file containing node features, edge relations, and node labels and these three NumPy arrays are generated. The three NumPy arrays are converted to PyTorch tensor in the load_data() method. A self-loop is added to the edge tensor, which allows nodes to retain their features during message passing.
The dataset was divided into a training set, a validation set, and a test set using random sampling. The training set constitutes 70% of the data, validation set 15% and test set 15%. Where training set is used to train the model and update the model parameters. The validation set is used to adjust the hyperparameters to avoid overfitting. Test set is used to finally ensure the performance of the model on unknown data.

## Dependencies

Python
Pytorch
Matplotlib
Numpy
scikit-learn

## Visualisation and Results

![accuracy](/Users/zhangxiangxu/PatternAnalysis-2024/accuracy_over_epochs.png)






