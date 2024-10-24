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

During the training process, the model learns the optimal parameters by minimizing the cross-entropy loss, and also uses forward propagation and the Adam optimizer, among others, to update the parameters step by step. Finally use early stop to prevent model overfitting.

## Data Pre-Processing

The data is loaded from an .npz file containing node features, edge relations, and node labels and these three NumPy arrays are generated. The three NumPy arrays are converted to PyTorch tensor in the load_data() method. A self-loop is added to the edge tensor, which allows nodes to retain their features during message passing.
The dataset was divided into a training set, a validation set, and a test set using random sampling. The training set constitutes 70% of the data, validation set 15% and test set 15%. Where training set is used to train the model and update the model parameters. The validation set is used to adjust the hyperparameters to avoid overfitting. Test set is used to finally ensure the performance of the model on unknown data.

## Files and Usage
Make sure the computer has a GPU, so using a GPU can greatly increase the speed of computation.
- `dataset.py`  This file handles loading and processing the Facebook Large Page-Page Network dataset for a Graph Neural Network (GNN). It includes functions to split the dataset into training(70%), validation(15%), and test sets(15%). As well as functions and classes prepared for subsequent training and testing.
- `modules.py` This file uses PyTorch to define a multi-layer graph convolutional network (GCN) for semi-supervised multi-class node classification. I have created a GCN with 4 convolutional layers here.
- `train.py` This file is used to handle the training of multi-layer graph neural networks (GNN). It includes stopping training early based on validation loss, saving the best performing models, and plotting loss and accuracy plots over the training calendar.
- `predict.py` This file is used to evaluate the trained GNN, computing the accuracy on the validation and test sets. Finally, t-SNE visualized node embedding is used to compress the classification results of the model into a two-dimensional space.


1. Download the file facebook.npz and change the dataset path to your own storage path in the dataset.py file.
2. Run train.py to train the model. The final model parameters and various information will be stored locally in model.pth, and generate loss and accuracy images for each epoch.
3. Run predict.py, which will calculate the accuracy of the trained model on the validation and test sets, and generate a t-SNE visualization of the node embeddings.



## Dependencies

Python
Pytorch
Matplotlib
Numpy
scikit-learn

## Visualisation and Results
After training the model 99 times on the training set and early stop, we obtained an accuracy of 94.87% on the test set, at which point the accuracy growth leveled off.

![accuracy]([plot/accuracy_over_epochs.png](https://github.com/Xiangxu-66/PatternAnalysis-2024/blob/9eda198dc4c3cbac159af55d258d0929a9da9c9e/plot/accuracy_over_epochs.png))
The graph above shows the change in accuracy of the GCN model over the course of 99 training sessions. We can see that the accuracy of both the training and validation sets rises rapidly at the beginning and starts to rise slowly at about the 20th epoch. The accuracy of the validation set is very close to the training set, which means that the model is not overfitting the training set.

![loss]([loss_over_epochs.png](https://github.com/Xiangxu-66/PatternAnalysis-2024/blob/9eda198dc4c3cbac159af55d258d0929a9da9c9e/plot/loss_over_epochs.png))
The graph above shows the model's loss over 99 training episodes, and we can see from the image that the validation loss values start out very high but then quickly decrease and stabilize, and in the last few epochs the training loss and validation loss are about the same, further indicating that the model is not overfitting.

![visualized]([TSNE_Visualization.png](https://github.com/Xiangxu-66/PatternAnalysis-2024/blob/9eda198dc4c3cbac159af55d258d0929a9da9c9e/plot/TSNE_Visualization.png))
The graph above represents each Facebook page in the dataset. The data is compressed to 2 dimensions by t-SNE, and each category is represented by a different color, resulting in a total of 4 different colors. The graph shows good clustering and from the results it is clear that the model has an accuracy path of 95.55%, which indicates that the model has learned the feature information about the nodes based on their categories.




