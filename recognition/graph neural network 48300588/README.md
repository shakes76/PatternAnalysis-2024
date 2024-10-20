# **Multi-Class Node Classification on Facebook Dataset**
# Kangqi Wang, 48300588

## **Summary**

I used the [Facebook Large Page-Page Network](https://snap.stanford.edu/data/facebook-large-page-page-network.html) divided into 4 categories (including politicians, governmental organisations, television shows and companies) to carry out a **semi supervised multi-class node classification**. The dataset consists of Facebook pages as nodes, features extracted as **128-dimensional vectors**, and edges representing mutual likes between pages.
My final results used **TSNE** embedding plot with ground truth in colours.

## **Algorithm Description**

The implemented GNN model is based on the Graph Convolutional Network (GCN) architecture proposed by [Kipf and Welling (2017)](https://arxiv.org/abs/1609.02907). The model consists of multiple graph convolutional layers that aggregate information from a node's neighbours, allowing it to capture both local and global graph structure.

Firstly, I used a **Custom Graph Convolution Layer**(`CustomGCNConv`) to implement the core GCN propagation rule:

$$
H^{(l+1)} = D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)}
$$
where $A$ is the adjacency matrix with added self-loops,  $D$ is the degree matrix, and $W$ is a trainable weight matrix of size `in_channels x out_channels`.

The second part is **GNN Model**(`GNNModel`), which is a complete Graph Neural Network model that stacks three `CustomGCNConv` layers to extract node features progressively through multiple graph convolutions.

## **How It Works**

The first thing to do is **pre-process** the data through the code in `dataset.py` as follows:

1. Load the Facebook dataset containing node features, edges, and labels.
2. Encode labels to **numerical values** using `LabelEncoder`.
3. Add **self-loops** to the adjacency matrix to include a node's own features during aggregation.
4. Split the dataset into **training (70%), validation (15%), and testing (15%)** sets.

After splitting the data, the model is trained using `train.py`:

1. **Initialise** the GNN model with specified input dimensions and hyper-parameters.
2. Use **Stochastic Gradient Descent (SGD)** optimiser with momentum and a learning rate scheduler for training.
3. Implement **early stopping** based on validation loss to prevent over-fitting.
4. Train the model over **multiple epochs**, recording losses and accuracy.

Finally, we just need to evaluate the model on a **test set** and calculate the accuracy.

## **Model Details**
 
 In `model.py`
- I created a custom GCN layer that manually computes the **propagation rule**. Also, the layer adds **self-loops** to the adjacency matrix and computes the **normalisation** using the degree matrix. These manual implementation gives this model more flexibility.
- What's more, I also added **batch normalisation layers** (`nn.BatchNorm1d`) after each convolutional layer to stabilise and accelerate training. After each convolutional layer, **ReLU activation and dropout** was applied to introduce non-linearity and prevent over-fitting.

 In `dataset.py`
- The dataset is splitted into 3 parts, including **training (70%), validation (15%), and testing (15%)**. Allocating 70% of the data to training set increases the diversity of examples the model sees, which can improve its ability to generalise. The validation set is used to tune hyper-parameters such as learning rate, momentum, and the number of layers without biasing the model to the test set. The test set verifies that the model generalises well and that the performance improvements are not just due to over-fitting to the training or validation sets. By keeping the test set separate and untouched during training and validation, we **prevent any inadvertent data leakage** that could inflate performance metrics.
- I used `np.random.shuffle` to **shuffle the indices** before splitting, ensuring a random distribution of nodes across train, validation, and test sets. This makes that the distribution of classes is approximately even across splits.

 In `train.py`
- I implemented **early stopping** based on validation loss to prevent over-fitting.
- Unlike adaptive optimiser like Adam, **SGD** provides more explicit control over the **learning rate and momentum**. Additionally, after I combined with techniques like momentum (`momentum=0.9`) and learning rate scheduling, SGD can converge as fast as or even faster than adaptive methods while maintaining good generalisation.
- **Cross-Entropy Loss** is ideal for multi-class classification problems, and can encourage the model to output accurate probability distributions.

In `predict.py`
- I used my **student number** as the random state(`random_state=48300588`) to ensure the reproducibility of the results. Without setting a `random_state`, t-SNE may produce different embedding each time it's run due to its inherent randomness.

