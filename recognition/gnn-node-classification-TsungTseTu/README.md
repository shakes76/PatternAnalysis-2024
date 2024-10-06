# GNN Node Classification on Facebook Large Page-Page Network

## 1. Project Overview
This project implements a Graph Neural Network (GNN) for semi-supervised multi-class node classification using the Facebook Large Page-Page Network Dataset. The goal of the project is to predict the class of each node (page) based on 128-dimensional feature vectors and the graph structure connecting the nodes.

Additionally, the node embeddings learned by the GNN model are visualized using t-SNE to demonstrate how well the model clusters nodes of the same class.

## 2. Requirements
To install the newest libraries, run:
```bash
pip install torch torch-geometric numpy matplotlib scikit-learn
```
### Dependencies
- Python: 3.8.10
- PyTorch: 2.4.1
- torch-geometric: 2.6.1
- NumPy: 1.26.4
- Matplotlib: 3.8.4
- scikit-learn: 1.5.1

Run with these specific versions I used to ensure reproducibility:
```bash
pip install torch==2.4.1 torch-geometric==2.6.1 numpy==1.26.4 matplotlib==3.8.4 scikit-learn==1.5.1
```

## 3. Model Structure
The implemented model is a multi-layer **Graph Convolutional Network (GCN)** using two `GCNConv` layers:
- **Input Layer**: 128-dimensional input features per node.
- **Hidden Layer**: Four layers with 128 hidden units and 4 attention heads.
- **Output Layer**: Provides logits for multi-class classification.


Hyperparameters (Best-Performing, accuracy:57.37%):
Optimizer: AdamW
Learning Rate: 0.001
Weight Decay: 0.0001
Hidden Dimensions: 128
Number of Layers: 4
Dropout: 0.2
Batch Normalization: Applied after each layer

## 4. How Model Works

My model implements a **Graph Attention Network (GAT)** for **semi-supervised node classification**. The model uses the graph structure and node features to classify nodes into predefined categories. Here's a brief overview of how the model works:

### Node Features and Graph Structure:
- Each node in the graph has a **128-dimensional feature vector** representing its attributes (e.g., properties of a Facebook page).
- The graph is represented by its **edges**, which define the connections between nodes (e.g., links between Facebook pages).

### Graph Attention Mechanism:
- The model uses a **Graph Attention Network (GAT)**, which computes attention scores for each node's neighbors.
- These attention scores determine how much **influence** a neighbor should have on a node’s representation. This allows the model to focus on the most important neighbors when updating each node’s representation.

### Multi-Layer Architecture:
- The GAT model consists of **four layers**, where each layer updates the node representations by aggregating information from neighboring nodes.
- After the final layer, the model outputs a **classification** for each node.

### Learning Process:
- The model is trained to **minimize the classification error** on the labeled nodes. The training process uses **AdamW** optimizer with a learning rate of **0.001** and a weight decay of **0.0001**.
- We also use a **learning rate scheduler** and **early stopping** to avoid overfitting and help the model converge faster.

### Evaluation:
- The model is evaluated on a separate test set, and its performance is visualized using **t-SNE** to project the high-dimensional node embeddings onto a 2D space.

## 5. Run the Code

### 5.1. Train the Model
The training script (`train.py`) trains the GNN model and saves the trained model weights in `gnn_model.pth`. You can run the training script as follows:



```bash
python train.py
```

This script:
- Load the **Facebook Large Page-Page Network Dataset**
- Initialize a GNN model with **128 input features** and **128 hidden layers**
- Random Seed: 42 for reproducibility
- Training is done with early stopping and learning rate scheduling (ReduceLROnPlateau).
- Train the GNN model for a number of epochs
- Save the trained model to *gnn_model.pth* for later prediction

### 5.2. Prediction and Evaluation
To evaluate the model and visualize the learned embeddings, run the prediction script (predict.py):

```bash
python predict.py
```

This script:
- Loads the pre-trained model.
- Predicts node classes and computes the accuracy on the test set.
- Visualizes node embeddings using t-SNE.

## 6. Dataset
The dataset I used is  **Facebook Large Page-Page Network dataset**, where the features are in the form of 128 dim vectors.
It consist of:
- Nodes: Facebook pages.
- Edges: Links between pages.
- Features: 128-dimensional vectors representing the features of each page.
- Labels: The class of each node (page).


### Preprocessing and Data Splitting

#### Preprocessing:
The original dataset was preprocessed by:
- **Loading the `.npz` file** and extracting the relevant arrays:
  - **edges**: A list of edges between nodes.
  - **features**: A matrix of 128-dimensional node features.
  - **target**: The class label for each node.
  
The features (128-dimensional vectors) are already preprocessed and ready to be used by the GNN model. No further feature transformation is applied; the dataset is ready to be fed directly into the GNN model. Tested without train/test split and got 97% accuracy, the preprocessed dataset is good.

#### Data Splitting:
The dataset is split into:
- **Training Set (80%)**: Used to train the model.
- **Test Set (20%)**: Used for final evaluation after training to assess model generalization.

The split is done using the train_test_split function from scikit-learn, with a fixed random seed (42) for reproducibility.


## 7. Result

### 7.1. Accuracy

The model achieved an accuracy of 57.37% on the test set after using the best hyperparameters (listed above).


### 7.2. Embedding Visualization (t-SNE)

Below is an ideal t-SNE visualization of the learned node embeddings which I ran without train-test split. Each color represents a different class, and the clusters indicate that the GNN effectively learned representations that separate nodes of different classes.

![SNE(ideal)](./images/SNE%20visualization%20(ideal).png)

**Ideal t-SNE Visualization** (First Image):
- The ideal t-SNE visualization shows distinct clusters, each representing a different class.
- Each color represents a different node class, and the clear separation between the clusters indicates that the model has successfully learned meaningful representations of the data, where nodes of the same class are grouped closely together.
- This ideal scenario implies that the embeddings generated by the model contain enough information to differentiate between the classes with minimal overlap or confusion.

Below is the t-SNE visualization of the learned node embeddings with train-test split.
![SNE(reality)](./images/SNE%20visualization%20(reality).png)

**Reality t-SNE Visualization** (Second Image):
- In the reality t-SNE visualization, the separation between clusters is not as clear as in the ideal case.
- There is more overlap between points of different colors (classes), indicating that the model's learned representations are not as distinct or discriminative as what we expected.
- Some nodes belonging to different classes are intermixed, meaning the model has difficulty separating some classes in the feature space, which could be:
  - Limited complexity in the model's architecture.
  - Insufficient feature information in the dataset itself.
  - The inherent difficulty of the task and noise in the data.

## 8. Drawbacks and Future Improvement

### 8.1. Model Performance and Scalability
- **Accuracy Ceiling**: The highest accuracy obtained was 57.37%, despite several hyperparameter tuning efforts. Achieving higher accuracy with this dataset may require more advanced techniques or larger hidden dimensions, but by doing this also increase training time significantly.

- **Training Time**: Increasing the number of hidden layers and hidden dimensions led to much longer training times without significant improvements in accuracy. Increases in model complexity beyond a certain point diminished returns in terms of performance.

### 8.2. Hyperparameter Tuning
- **Limited Gains from Additional Layers**: Adding more layers (beyond four) resulted in diminishing accuracy improvements and sometimes worsened the performance due to overfitting or vanishing gradients.

- **Learning Rate Sensitivity**: The learning rate required careful tuning. Lower learning rates slowed convergence significantly, while higher rates caused instability during training.

- **Grid Search**: Grid Search with the given hyperparameter would take hours to run one training, and also without significant output.


## 9. Reference
1. Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks. arXiv preprint arXiv:1609.02907.
2. Facebook Large Page-Page Network Dataset: [Data Source](https://snap.stanford.edu/data/facebook-large-page-page-network.html), and the [Paritally proecessed one](https://graphmining.ai/datasets/ptg/facebook.npz)

## 10. License
This project is licensed under the Apache License - see the [License](../../LICENSE) file for details.