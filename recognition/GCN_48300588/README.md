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

## **Usage**

1. Make sure your computer has a GPU.
2. Download the dataset and change the path to load the data in `dataset.py`
3. Run the `train.py` to train the model. This will save the best model as `best_model.pth` and generate loss and accuracy plots saved as `loss.png` and `accuracy.png`.
4. Make sure the `random_state=48300588` is not modified, and run the `predict.py` to evaluate the model and visualise embedding. This will generate a t-SNE plot of node embedding saved as `tsne_embeddings.png`, and print sample predictions and test accuracy.


## **Hyper-parameters Comparison**

Learning Rate:

$$\begin{array}{|c|c|c|}
\hline
\text{Learning Rate (lr)} & \text{Early Stopping Epoch} & \text{Test Accuracy} \\
\hline
0.7 & 154 & 0.9395 \\
0.5 & 147 & 0.9487 \\
0.3 & 160 & 0.9365 \\
0.1 & 158 & 0.9293 \\
0.05 & 163 & 0.9050 \\
\hline
\end{array}
$$

Weight Decay:

$$
\begin{array}{|c|c|c|}
\hline
\text{Weight Decay} & \text{Early Stopping Epoch} & \text{Test Accuracy} \\
\hline
7 \times 10^{-4} & 179 & 0.9318 \\
5 \times 10^{-6} & 165 & 0.9353 \\
5 \times 10^{-4} & 147 & 0.9487 \\
5 \times 10^{-3} & 31 & 0.9101 \\
3 \times 10^{-4} & 110 & 0.9350 \\
1 \times 10^{-4} & 175 & 0.9344 \\
\hline
\end{array}
$$

Momentum:

$$
\begin{array}{|c|c|c|}
\hline
\text{Momentum} & \text{Early Stopping Epoch} & \text{Test Accuracy} \\
\hline
0.99 & 18 & 0.9023 \\
0.95 & 147 & 0.9487 \\
0.9 & 110 & 0.9341 \\
0.85 & 111 & 0.9294 \\
0.8 & 119 & 0.9297 \\
\hline
\end{array}
$$

Gamma:

$$
\begin{array}{|c|c|c|}
\hline
\text{Gamma} & \text{Early Stopping Epoch} & \text{Test Accuracy} \\
\hline
0.7 & 248 & 0.9306 \\
0.5 & 147 & 0.9487 \\
0.3 & 158 & 0.9244 \\
0.1 & 178 & 0.9256 \\
0.01 & 117 & 0.9267 \\
\hline
\end{array}
$$

Data Size:

$$\begin{array}{|c|c|c|c|c|}
\hline
\text{Train Size} & \text{Val Size} & \text{Test Size} & \text{Early Stopping Epoch} & \text{Test Accuracy} \\
\hline
0.6 & 0.2 & 0.2 & 134 & 0.9326 \\
0.7 & 0.15 & 0.15 & 147 & 0.9487 \\
0.8 & 0.1 & 0.1 & 141 & 0.9395 \\
\hline
\end{array}
$$


My final hyper-parameter values:

$$
\begin{array}{|c|c|}
\hline
\text{Hyperparameter} & \text{Value} \\
\hline
\text{Learning Rate} & 0.5 \\
\text{Weight Decay} & 5 \times 10^{-4} \\
\text{Momentum} & 0.95 \\
\text{Gamma} & 0.5 \\
\text{Train Size} & 0.7 \\
\text{Val Size} & 0.15 \\
\text{Test Size} & 0.15 \\
\hline
\text{Early Stopping Epoch} & 147 \\
\text{Test Accuracy} & 0.9487 \\
\hline
\end{array}
$$


## **Results and Visualisation**

Results of the training set in the last 10 epochs:

$$
\begin{array}{|c|c|c|c|c|}
\hline
\text{Epoch} & \text{Loss} & \text{Val Loss} & \text{Train Acc} & \text{Val Acc} \\
\hline
138 & 0.1803 & 0.1961 & 0.9578 & 0.9421 \\
139 & 0.1768 & 0.1968 & 0.9576 & 0.9398 \\
140 & 0.1796 & 0.1971 & 0.9580 & 0.9398 \\
141 & 0.1751 & 0.1985 & 0.9572 & 0.9383 \\
142 & 0.1748 & 0.1986 & 0.9570 & 0.9392 \\
143 & 0.1734 & 0.1967 & 0.9571 & 0.9407 \\
144 & 0.1780 & 0.1951 & 0.9583 & 0.9407 \\
145 & 0.1732 & 0.1955 & 0.9581 & 0.9404 \\
146 & 0.1748 & 0.1957 & 0.9585 & 0.9407 \\
147 & 0.1700 & 0.1961 & 0.9585 & 0.9409 \\
\hline
\end{array}
$$

$$
\text{Early stopping at epoch 147}
$$

$$
\text{Test Accuracy: } 0.9487
$$

Results of the test set:

$$
\begin{array}{|c|c|c|}
\hline
\text{Node} & \text{Predicted Label} & \text{True Label} \\
\hline
0 & 0 & 0 \\
1 & 2 & 2 \\
2 & 1 & 1 \\
3 & 2 & 2 \\
4 & 3 & 3 \\
5 & 3 & 3 \\
6 & 3 & 3 \\
7 & 3 & 3 \\
8 & 2 & 2 \\
9 & 2 & 2 \\
\hline
\end{array}
$$

Figure 1 Training and Validation Accuracy:

![accuracy.png](PatternAnalysis-2024/recognition/accuracy.png)

Figure 2 Training and Validation Loss:

![loss.png](PatternAnalysis-2024/recognition/loss.png)

Figure 3 t-SNE Visualisation of Node Embedding:

![tsne_embeddings.png](PatternAnalysis-2024/recognition/tsne_embeddings.png)







