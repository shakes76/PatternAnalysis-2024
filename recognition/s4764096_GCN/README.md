# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion
Graph Convolutional Network for Facebook Page Classification
This project implements a Graph Convolutional Network (GCN) to classify Facebook pages into four categories: Politicians, Government Organizations, Television Shows, and Companies. The classification task is based on a Facebook Page-Page network dataset where nodes represent Facebook pages, edges represent mutual likes, and features are page descriptions.

Project Overview
The task is to perform semi-supervised multi-class node classification using a GCN. The input graph consists of pages as nodes, mutual likes as edges, and 128-dimensional feature vectors representing each page's description. The goal is to classify each page into one of the four predefined categories.

Tasks Involved
This project focuses on the following key tasks:

Graph Node Classification: Using GCN to classify nodes (Facebook pages) based on their features and connections.
Visualization: UMAP (Uniform Manifold Approximation and Projection) is used to reduce the dimensionality of the node embeddings and visualize the classification results.
Data Description
The dataset used in this project is preprocessed and provided in the form of a .npz file containing:

edges: A 171,002 x 2 matrix representing the mutual likes (connections) between Facebook pages (nodes).
features: A 22,470 x 128 matrix, where each row represents a 128-dimensional feature vector for a Facebook page.
target: A 22,470-length array representing the class labels for each page, with values ranging from 0 to 3, corresponding to the four categories.
Data Splitting
80% of the data is used for training.
10% is used for testing.
10% is used for validation.
Model Architecture
The Graph Convolutional Network (GCN) used in this project has the following architecture:

Graph Convolutional Layers: Three GraphConv layers, which propagate information between connected nodes.
Dropout Layers: Dropout is applied after each GraphConv layer to prevent overfitting.
ReLU Activation: Each GraphConv layer uses ReLU as the activation function to introduce non-linearity.
Linear Classifier: A final fully connected layer classifies the nodes into one of four categories.
Detailed Architecture:

Input dimension: 128
Hidden layers: 3 GraphConv layers with dimensions [128, 128, 256, 256]
Final output dimension: 4 (number of classes)
Training Details
Hyperparameters
Learning Rate: 0.001
Optimizer: Adam with weight decay of 5e-4
Epochs: 100
Dropout Rate: 0.5
Loss Function: Cross-Entropy Loss
Training Process
During the training process, the model learns to classify nodes based on their features and neighbors. The following metrics are tracked:

Training Loss: Measures how well the model fits the training data.
Test Accuracy: Measures the model's performance on the test set.
At the end of training, a UMAP plot is generated to visualize the learned embeddings of the nodes.

Training Results
The model was trained for 100 epochs, and the training loss and test accuracy over time are shown below. In the final few epochs, the loss stabilizes and the test accuracy reaches around 92.85%.

Example training output:

Epoch 100/100, Loss: 0.2588612735271454, Test Accuracy: 0.9285714030265808
UMAP Visualization
After training, UMAP is applied to the learned node embeddings to reduce them to 2D for visualization purposes. The UMAP plot shows how well the model has learned to separate the nodes into different classes.


The different colors in the UMAP plot represent the four classes, with each point corresponding to a Facebook page.

Files Overview
dataset.py: Contains functions for loading and preparing the data. It uses DGL to create the graph and prepare the training/test masks.
modules.py: Defines the GCN model architecture, including GraphConv layers, ReLU activation, and dropout.
train.py: The main training script, which trains the GCN model, tracks the loss and accuracy, and generates the UMAP plot for visualization.
predict.py: Loads the trained model and visualizes the results using UMAP.
Trained Model: By default, the model weights are saved in the _pycache_ directory under the name gcn_model.pth.
Running the Project
Training the Model
To train the model, use the following command:

python train.py
This will train the GCN on the Facebook Page-Page network and generate the UMAP visualization.

Visualizing the Results
Once the model is trained, you can visualize the node embeddings using UMAP by running:

python predict.py
This will load the trained model and generate a UMAP plot showing how well the model has classified the nodes into the four categories.

Dependencies
The following Python packages are required to run the project:

torch: For building and training the GCN model.
dgl: For creating and manipulating graph data structures.
numpy: For handling numerical data.
matplotlib: For plotting training metrics and UMAP visualizations.
umap-learn: For reducing the dimensionality of the learned node embeddings.