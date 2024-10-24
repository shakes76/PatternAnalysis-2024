# PageGCN: Semi-Supervised Node Classification for Facebook Pages

## Data background
This project uses the [Facebook Large Page - Page Network dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html) to perform semi-supervised multi-class node classification using a graph convolutional network (GCN). The dataset consists of connections between Facebook pages, which are divided into the following four categories: (0) Politicians, (1) Governmental Organizations, (2) Television Shows, and (3) Companies.

## Dataset preprocessing
This section describes the preprocessing of the Facebook large-scale Page-Page network dataset for training a graph convolutional network (GCN) model. The preprocessing ensures that the input features, labels, and adjacency matrices are properly formatted and normalized to be compatible with the GCN model.

### Data loading
The dataset is stored in a compressed .npz file and loaded via numpy. The dataset contains:
- **Edges**: An array representing the connection relationship between nodes in the graph (i.e., edges).
- **Features**: The feature matrix of all nodes, each row represents the feature vector of a node.
- **Target**: The category label of each node, used for multi-category classification tasks.

### Data spliting
The nodes are randomly divided into training set, validation set and test set, and the division ratio is as follows:
- **Training Set:** 70% of the nodes, a total of 15,728 nodes.
- **Validation Set:** 15% of the nodes, a total of 3,370 nodes.
- **Test Set:** 15% of the nodes, a total of 3,372 nodes.

The nodes are randomly indexed to ensure good distribution among the datasets. Masks are generated for each dataset to identify the nodes belonging to each set during training and evaluation.

### Adjacency matrix preprocessing
The graph structure is defined by the edges between nodes and represented as an adjacency matrix:
- **Add Self-loops:** Add self-loops to each node to ensure that each node can consider its own characteristics during message passing.
- **Normalization:** The adjacency matrix is ​​normalized using the symmetric normalization method and adjusted based on the node degree. This can prevent high-degree nodes from excessively affecting the calculation results during training and ensure stable training.
- **Sparse Tensor Representation:** The normalized adjacency matrix is ​​converted to a PyTorch sparse tensor to facilitate efficient memory utilization during model training.

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

## Model Design
The GCN model consists of three layers of graph convolutional layers (GCN Layer), and batch normalization and dropout are applied after each layer to prevent overfitting. The output of the last layer is normalized by the Softmax function for classification.

The output of each layer of GCN can be expressed as: 

$$
Z = \text{ReLU}\left( \hat{A} X W \right)
$$

### Model components
- **First GCNLayer**: The input features have a dimension of in_channels and the output features have a dimension of hidden_channels. We apply batch normalization and ReLU activation function after the output, and introduce dropout to prevent overfitting.
- **Second GCNLayer**: This layer is similar to the first layer, with an input dimension of hidden_channels and an output dimension of hidden_channels. Batch normalization, ReLU activation, and dropout are also applied.
- **Third GCNLayer**: The output layer has a dimension of hidden_channels to num_classes (number of categories), and this layer does not contain an activation function. The output is used for classification tasks through log-softmax.

### Model Regularization
- **Batch Normalization**: Batch Normalization is applied after each layer to speed up the convergence of the model and stabilize the training process. It normalizes the output data of each batch to make its mean 0 and variance 1.
- **Dropout**: After the output of each layer, we use dropout (random inactivation) to discard the output of some neurons with a certain probability (controlled by the parameter dropout_prob), thereby enhancing the generalization ability of the model and avoiding overfitting.

### Forward propagation of the model
- **First GCN layer**: linearly transform the input feature matrix, propagate features through the adjacency matrix, and then apply batch normalization, ReLU activation, and dropout.
- **Second GCN layer**: the same process, first feature transformation, adjacency matrix propagation, then batch normalization, ReLU activation, and dropout.
- **Third GCN layer**: the output layer does not perform activation function operations and directly outputs category scores.
- **Softmax**: after the last layer, the model uses the log-softmax function for normalization, so that the output becomes a probability distribution of each category.

### Loss Function
The loss function of the model uses cross-entropy loss, which is suitable for multi-category classification problems. During training, we optimize the model parameters by minimizing this loss function.

## Training results and visualization
### Training part
#### Main steps:
- **Data loading and preprocessing**: Load node features, adjacency matrix and labels from the Facebook page network dataset, and divide them into training, validation and test sets.
- **GCN model initialization**: Initialize the GCN model according to the number of node features, hidden layer size and number of categories.
- **Forward propagation**: Input node features and adjacency matrix, and calculate the classification results of each node through the model.
- **Loss calculation**: Use the cross entropy loss function to calculate the training and validation losses of the model.
- **Backward propagation and gradient update**: Calculate the gradient through back propagation, and use the Adam optimizer to update the model parameters. Use gradient clipping to prevent gradient explosion.
- **Early stopping mechanism and learning rate scheduling**: During the training process, if the validation set accuracy no longer improves, enable the early stopping mechanism. At the same time, use the learning rate scheduler to adjust the learning rate according to the validation set performance.
- **Performance evaluation**: During the training process, record and output the training loss, validation loss, training accuracy and validation accuracy of each epoch, and evaluate the test set accuracy at the end.
- **Visualization**: Draw and save the accuracy and loss change curves on the training and validation sets.

#### Output:
The loss and accuracy of each epoch during model training, as well as the final test set accuracy.
The accuracy and loss change curve of the model.

![GCN_accuary.png](https://github.com/YuchengWatoo/PatternAnalysis-2024/blob/1ecb8cdc14275c70d1ce1c366ca58aa647ad1f80/Multi-layer_GNN_47914111/plots/GCN_accuracy.png)

In the first 20 rounds, the training accuracy and validation accuracy increased rapidly, indicating that the model effectively learned the graph structure and features at the beginning.
In the subsequent training, the accuracy gradually stabilized and was relatively stable at about 60 rounds.
The accuracy curves of the training set and the validation set are very close, both close to 0.9. This shows that the performance on both the training set and the validation set is good and there is no overfitting.

![GCN_loss.png](https://github.com/YuchengWatoo/PatternAnalysis-2024/blob/1ecb8cdc14275c70d1ce1c366ca58aa647ad1f80/Multi-layer_GNN_47914111/plots/GCN_loss.png)
In the first 20 rounds, both the training loss and the critical loss dropped rapidly, indicating that the model quickly optimized the parameters and reduced the error at the beginning.
In the subsequent training, the loss value gradually stabilized and was relatively stable at about 80 rounds.
The loss curves of the training set and the validation set are very close. The validation loss is slightly lower than the training loss, indicating that the model has good generalization ability on the validation set.

These two figures show that the GCN model performs well during training, and the model achieves high accuracy and low loss values ​​on both the training and validation sets. There is no obvious overfitting or underfitting phenomenon.

Save the trained GCN model (GCN_Model.pt).

### Prediction part
#### Main steps:
- **Data loading and preprocessing**: Load node features, adjacency matrix and labels to ensure consistency with the data used in training.
- **Model loading**: Load the previously trained GCN model and move it to the computing device (CPU or GPU).
- **Forward propagation**: Input node features and adjacency matrix, calculate the output representation of the node through the model for visualization.
- **t-SNE dimension reduction**: Use the t-SNE algorithm to reduce high-dimensional node features to two dimensions, which is convenient for visualizing the distribution relationship between nodes.
- **Visualization**: Draw and save a scatter plot of nodes in two-dimensional space, and colors represent different node categories.

#### Output:
The node feature visualization diagram after dimensionality reduction using the t-SNE algorithm shows the node classification representation learned by the model, and saves the diagram as an image file.
![TSNE_train_plot.png](https://github.com/YuchengWatoo/PatternAnalysis-2024/blob/1ecb8cdc14275c70d1ce1c366ca58aa647ad1f80/Multi-layer_GNN_47914111/plots/TSNE_train_plot.png)

The boundaries between the four categories are relatively clear, indicating that the model has successfully separated the features of different categories. In particular, 1 and 3 show obvious separation. Category 2 shows a certain degree of aggregation, indicating that the node features of this category are relatively similar.
Although most points show clear classification, there is confusion at the junction of 1 and 2. This shows that the feature similarity between nodes is high.

## Hyperparameters and Functions
- **Epochs:** 100
Set to 100 iterations to ensure that the model has enough training time to converge, and cooperate with the early stopping mechanism to avoid overfitting.
- **Learning Rate:** 0.01
Use a moderate learning rate to balance the convergence speed and stability. Too high a learning rate may cause gradient explosion or oscillation, while too low a learning rate will make the model train slowly.
- **Hidden Dimensions:** 16
The hidden dimension is set to 16 as a suitable balance point to capture the high-order features of the data while avoiding excessive computational overhead.
- **Dropout Probability:** 0.5
Set to 50% dropout probability to prevent the model from overfitting during training.
- **Gradient Clipping:** 1.0
Clip the gradient at each backpropagation to prevent the gradient explosion problem. By limiting the maximum norm of the gradient, the training process is ensured to be more stable.
- **Optimizer:** Adam
The Adam optimizer is used, and its adaptive learning rate adjustment mechanism helps improve the training efficiency of the model while maintaining high accuracy.
- **Loss Function:** Cross Entropy Loss
Cross Entropy Loss is a standard loss function for multi-class classification tasks, which is used to measure the difference between the classification probability output by the model and the actual category.
- **Early Stopping:**
Set the patience value to 10 rounds. If the accuracy of the validation set does not improve within 10 rounds, stop training to prevent overfitting.

## Usage
1. Download the dataset and change the path to load the data in dataset.py
2. Run the train.py to trian the model and plot the results of accuracy and loss. Also save model as GCN_model.pt.
3. Run the predict.py, visualise tsne plot and saved as TSNE_train_plot.png.
## Dependencies
Required packages are as follows:
- torch 2.4.1
- sklearn 1.3.0
- scipy 1.10.1
- matplotlib 3.7.2
- numpy 1.24.3
## Reference