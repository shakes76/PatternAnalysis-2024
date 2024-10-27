# Semi Supervised Multi-Class Node Classification using a Multi-Layer Graph Neural Network

Author: Anthony Ngo
Student Number: 48019022

## Project Overview
This project implements a variety of Graph Neural Network (GNN) architectures for node classification, specifically for the Facebook Large Page-Page Network dataset. The goal of the project was to sufficiently classify nodes of the graph data into 4 categories: politicians, governmental organisations, television shows and companies. 

A potential use of this model would be to improve content recommendations for Facebook users, or categorise and moderate Facebook pages.

## Table of Contents
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Inference](#inference)
- [References](#references)

## Project Structure
The following files are included in the repository:
- `train.py`: Training and evaluation logic wrapped in a main training loop.
- `modules.py`: Contains various GNN architectures, defines layers and forward passes.
- `dataset.py`: Custom dataloader for the Facebook graph data.
- `predict.py`: Script for running inference on the dataset.
- `plotting.py`: Script for plotting the TSNE embeddings for a select model.
- `wandb_config.py`: Script for initialising Weights and Biases logging.
- `utils.py`: (Unused)

## Dependencies
This project requires the following libraries:
- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- PyTorch Geometric
- PyTorch Sparse
- NumPy
- Matplotlib
- Scikit-learn
- Weights and Biases (wandb)

You can install the required packages using pip:
```bash
pip install torch torchvision torchaudio torch-geometric numpy matplotlib scikit-learn wandb torch-sparse
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anthonylastnamengo/PatternAnalysis-2024/tree/topic-recognition
   ```

2. Install the dependencies as mentioned above.

## Data Preparation
The dataset can be obtained from [this link](https://graphmining.ai/datasets/ptg/facebook.npz) in a partially processed format, where the features are represented as 128-dimensional vectors. After downloading, save the dataset in the following directory structure:
```bash
PATTERNANALYSIS-2024/
    recognition/
        48019022_GNN/
            facebook.npz
```
In the dataloader, graph data indexes are split in a 80/10/10 split across training, validation and testing indexes.

## Usage
The repository implements 4 GNN architectures:
- Graph Convolutional Network (**GCN**)
- Graph Attention Network (**GAT**)
- GraphSAGE (**SAGE**)
- Simple Graph Convolution (**SGC**)

1. To select a model to train, change [this line](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/b1280aff8f3637526ee9a34e0b542718c09f1e08/recognition/48019022_GNN/train.py#L128) in `train.py` to one of the above bracketed model types. If you wish to reproduce data splits, set a seed with [this line](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/b1280aff8f3637526ee9a34e0b542718c09f1e08/recognition/48019022_GNN/train.py#L130).

2. To train the model, run the following command:
   ```bash
   python train.py
   ```

4. To run inference on the trained model, first [select the model](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/d62be4722f58923f5a8fb3dc5e877f17e40f907b/recognition/48019022_GNN/predict.py#L19) type trained in predict.py, [set a seed](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/d62be4722f58923f5a8fb3dc5e877f17e40f907b/recognition/48019022_GNN/predict.py#L14), then run:
   ```bash
   python predict.py
   ```

5. To visualise the embeddings, again, [change the model type](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/d62be4722f58923f5a8fb3dc5e877f17e40f907b/recognition/48019022_GNN/plotting.py#L37) and [seed](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/d62be4722f58923f5a8fb3dc5e877f17e40f907b/recognition/48019022_GNN/plotting.py#L32) in `plotting.py`, then execute:
   ```bash
   python plotting.py
   ```
   This will generate a TSNE plotting of the model's clustering.

## Model Architecture
As state above, this project implements the following GNN architectures:
- **Graph Convolutional Network** (GCN)
- **Graph Attention Network** (GAT)
- **Graph Sample and Aggregation** (Graph SAGE)
- **Simple Graph Convolution** (SGC)

Each model has been encapsulated in the `modules.py` file, with a defined forward pass and relevant layers. The next sections contain brief descriptions of how each model functions.

### Graph Convolutional Networks (GCN)
Graph Convolutional Networks (GCN) leverage the principles of convolutional neural networks to operate directly on graph-structured data. The key idea is to use the spectral domain of the graph to learn a transformation of node features. GCNs are designed to perform node classification tasks by aggregating features from a node's neighbourhood in the graph, effectively capturing local structures. 

GCNs are particularly effective for semi-supervised learning, where only a small subset of nodes has labeled information.

The GCN architecture implements:
- 2 layers of GCNConv
- ReLU activation in between
- dropout layer for regularisation, and applied to prevent overfitting.

### Graph Attention Networks (GAT)
Graph Attention Networks (GAT) introduce an attention mechanism to the graph convolution process, allowing nodes to weigh their neighbours' contributions based on their importance. GATs utilise self-attention to learn the coefficients for each edge, thereby adapting the influence of connected nodes dynamically. 

By focusing on relevant neighbours, GATs enhance the model's expressiveness and robustness against noise in the graph structure.

The GAT architecture implements:
- 1st GATConv layer with multi-head attention (8 heads) to learn weighted relations
- ReLU activation layer
- A final GATConv layer to combine the attention head and project to the output classes (1 head)

### Graph Sample and Aggregation (GraphSAGE)
Graph Sample and Aggregation (GraphSAGE) is designed to handle large graphs by sampling a fixed-size neighbourhood of each node rather than considering the entire graph. It employs various aggregation functions (mean, LSTM, pooling) to merge information from sampled neighbours into the target node's representation. 

GraphSAGE enables inductive learning, allowing the model to generalise to unseen nodes during training.

The GraphSAGE architecture implements:
- 1st SAGEConv layer aggregates neighbourhood information and learns a hidden representation
- 2nd SAGEConv layer aggregates said hidden representations and projects them to the outputs
- Dropout is used to regularise
- ReLU is used to activate nodes in between first and second SAGEConv layers

### Simplified Graph Convolution (SGC)
Simplified Graph Convolution (SGC) reduces the complexity of graph convolutional operations by removing non-linearities and employing a single linear transformation. The key innovation is the simplification of the message-passing framework into a single matrix multiplication, effectively treating the graph structure as a single layer. 

SGC is efficient and effective for node classification, particularly in scenarios where deeper layers do not significantly improve performance. As the SGC architecture is relatively simple, this implementation was done without the use of the PyTorch Geometric built-in SGConv, and instead implements propagation and degree-based normalisation manually.

The SGC architecture implements:
- A k-step propagation layer for neighbourhood aggregation across k hops
- a linear transformation layer that projects aggregated features to output classes

## Training Process
### Data Splits
Before training, the data is prepared using a GNNDataLoader, which takes in numpy data from the file facebook.npz and divides it into three subsets:
- **80% Training Set**: Used to train the model and update weights.
- **10% Validation Set**: Used to monitor the model's performance during training in methods to alleviate overfitting.
- **10% Test Set**: Used to evaluate the final model performance after training.

The 80/10/10 split ratio is a common practice in machine learning.

### Model Initialisation
The model is selected based on specified architecture (GCN, GAT, GraphSAGE, SGC). Each model is instantiated with the following parameters:
- **Input dimensions**: The number of input features for each node (in this case, 128).
- **Hidden Dimension**: The number of neurons in the hidden layer (set to 64).
- **Output Dimension**: The number of output classes, which is derived from the maximum label in the dataset (in this case, 4 classes).

### Training Loop
The core of the training process is in the training_loop function, which iterates over a predefined number of epochs to train the model. Here are the detailed steps involved:

**Epoch Loop**: 
For each epoch, the following steps occur:
1. **Training Phase**:
   - The _train_model function is called to perform a forward pass and backpropagation.
   - The model’s predictions are computed, and the loss is calculated using the cross-entropy criterion.
   - Gradients are cleared, calculated, and the optimiser updates the model parameters based on these gradients.
     
2. **Validation and Testing**:
   - After training for the current epoch, the _evaluate_model function is called to assess the model’s performance on the validation set.
   - The validation loss is computed, and accuracy on the test set is evaluated by comparing the predicted and actual labels.
     
3. **Logging Metrics**:
   - The training loss, validation loss, test accuracy, and learning rate are logged to Weights and Biases (WandB) for tracking progress.
     
4. **Early Stopping**:
   - If the validation loss does not improve for a certain number of epochs (set by patience_lim), the training loop terminates early to avoid overfitting.
     
5. **Loss Plotting**:
   - After training, a plot of training and validation loss is generated to visually assess the training process.

**Saving the Model**

At the end of each epoch, if the validation loss improves, the model’s weights are saved. The weights can be restored later for evaluation or inference. The file is saved with a name that indicates the model architecture.

**Learning Rate Scheduling**

A learning rate scheduler (StepLR) is employed to adjust the learning rate during training. Every 50 epochs, the learning rate is multiplied by a gamma factor (0.5 in this case) to help the optimiser converge to better minima.

**Final Logging**

At the end of training, additional metrics such as total training time and the number of parameters in the model are logged to WandB for later analysis.

### Training Details
- Maximum epochs: **300**
- Optimiser: **Adam**
   - The Adam optimiser was chosen and used with weight decay for fast convergence.
      - Learning Rate (Max): 0.01
      - Weight Decay: 0.0005 (L2 Regularisation)
- Loss Function: **Cross Entropy Loss**
   - Cross Entropy Loss was used as a criterion due to its effectiveness for multi-class problems. CE minimises the loss by aligning its predicted probability distribution with the actual class distribution
- Scheduler: **StepLR**
   - The learning rate scheduler StepLR was used for improved convergence and stability, as well as its advantage of better exploration in early stages, allowing it to explore the parameter space more widely and escape local minima.

## Results
The primary metrics logged and used for the analysis of the GNN architectures were:

- **Test Accuracy**
    - After the model is trained, test accuracy provides insight into how well the model generalises to unseen data.
   
- **Training Loss**
    - Monitoring the training loss helps in understanding whether the model is learning or overfitting.

- **Validation Loss**
    - Validation loss is used to tune model hyperparameters, such as learning rate or regularisation techniques, and assess the model’s performance during training.

- **T-SNE Manifold Clustering**
    - Well-separated clusters in the t-SNE plot imply that the model has learned distinct representations for the different node categories, which indicates good classification performance.

### Testing Methodology
Each model was provided the same set of data. As the data is randomly split each run, a manual random seed was introduced to ensure reproducibility of results and consistency across models. The seeds are included in the `dataset.py` file. Each model was then tested on each seed for a total of 5 runs. The following graphs display the performances of each model.

### GCN Performance
**GCN Loss Plot:**

![GCN Loss Plot](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GCN_Train.png)

**GCN Validation Loss:**

![GCN Validation Loss](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GCN_Validation.png)

**GCN Test Accuracy:**

![GCN Test Accuracy](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GCN_Accuracy.png)

**GCN T-SNE Visualisation:**

![GCN TSNE](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GCN_TSNE.png)

### GAT Performance
**GAT Loss Plot:**

![GAT Loss Plot](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GAT_Train.png)

**GAT Validation Loss:**

![GAT Validation Loss](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GAT_Validation.png)

**GAT Test Accuracy:**

![GAT Test Accuracy](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GAT_Accuracy.png)

**GAT T-SNE Visualisation:**

![GAT TSNE](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/GAT_TSNE.png)

### GraphSAGE Performance
**GraphSAGE Loss Plot:**

![GraphSAGE Loss Plot](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SAGE_Train.png)

**GraphSAGE Validation Loss:**

![GraphSAGE Validation Loss](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SAGE_Validation.png)

**GraphSAGE Test Accuracy:**

![GraphSAGE Test Accuracy](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SAGE_Accuracy.png)

**GraphSAGE T-SNE Visualisation:**

![GraphSAGE TSNE](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SAGE_TSNE.png)

### SGC Performance
**SGC Loss Plot:**

![SGC Loss Plot](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SGC_Train.png)

**SGC Validation Loss:**

![SGC Validation Loss](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SGC_Validation.png)

**SGC Test Accuracy:**

![SGC Test Accuracy](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SGC_Accuracy.png)

**SGC T-SNE Visualisation:**

![SGC TSNE](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/SGC_TSNE.png)


### Model Comparisons
For the sake of the methodology, the main metric used to benchmark each architecture will be the training accuracy.

Below is the plot of each model's accuracy.

![Overall architecture accuracy](https://github.com/anthonylastnamengo/PatternAnalysis-2024/blob/topic-recognition/recognition/48019022_GNN/assets/Overall_Accuracy.png)

From the plot above, the architecture with the single highest accuracy (95.149%) on the testing data labels was the Graph Attention Network. This is theorised to be a result of the GAT's ability to focus on the most relevant nodes in the graph data using its attention mechanism.

The GAT also performed best on average:

Architecture  | Average Accuracy
------------- | -------------
GCN  | 93.0216%
GAT  |  **94.3392%**
GraphSAGE  | 92.95184%
SGC  | 886428%

## Inference
To test trained models on the graph data, use the `predict.py` script. This script will load a selected trained model and attempt to classify each node of the dataset.

## References
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ArXiv:1609.02907 [Cs, Stat]. https://arxiv.org/abs/1609.02907

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. ArXiv:1710.10903 [Cs, Stat]. https://arxiv.org/abs/1710.10903

- Hamilton, W. L., Ying, R., & Leskovec, J. (2018). Inductive Representation Learning on Large Graphs. ArXiv:1706.02216 [Cs, Stat]. https://arxiv.org/abs/1706.02216

- Wu, F., Zhang, T., Souza Jr. , A. H. de, Fifty, C., Yu, T., & Weinberger, K. Q. (2019, June 20). Simplifying Graph Convolutional Networks. ArXiv.org. https://doi.org/10.48550/arXiv.1902.07153