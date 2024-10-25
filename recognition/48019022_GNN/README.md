# Semi Supervised Multi-Class Node Classification using a Multi-Layer Graph Neural Network

Author: Anthony Ngo
Student Number: 48019022

## Project Overview
This project implements a variety of Graph Neural Network (GNN) architectures for node classification, specifically for the Facebook Large Page-Page Network dataset. The goal of the project was to sufficiently classify nodes of the graph data into 4 categories: politicians, governmental organizations, television shows and companies. 
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
- [Visualisations](#visualisations)
- [Performance Metrics](#performance-metrics)
- [References](#references)

## Project Structure
The following files are included in the repository:
- train.py: Training and evaluation logic wrapped in a main training loop.
- modules.py: Contains various GNN architectures, defines layers and forward passes.
- dataset.py: Custom dataloader for the Facebook graph data.
- predict.py: Script for running inference on the dataset. 
- plotting.py: Script for plotting the TSNE embeddings for a select model.
- wandb_config.py: Script for initialising Weights and Biases logging.
- utils.py: (Unused)

## Dependencies
This project requires the following libraries:
- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- PyTorch Geometric
- NumPy
- Matplotlib
- Scikit-learn
- Weights and Biases (wandb)

You can install the required packages using pip:
```bash
pip install torch torchvision torchaudio torch-geometric numpy matplotlib scikit-learn wandb
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anthonylastnamengo/PatternAnalysis-2024/tree/topic-recognition
   ```

2. Install the dependencies as mentioned above.

## Data Preparation
The dataset can be obtained from (https://graphmining.ai/datasets/ptg/facebook.npz) in a partially processed format where features are in the form of 128 dim vectors. 
After downloading, save the dataset in the following structure:
    ```bash
    PATTERNANALYSIS-2024/
        recognition/
            48019022_GNN/
                facebook.npz
    ```

## Usage
The repository implements 4 GNN architectures:
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE (SAGE)
- Simple Graph Convolution (SGC)

1. To select a model to train, change line 133 of train.py to one of the above bracketed model types.

2. To train the model, run the following command:
   ```bash
   python train.py
   ```

4. To run inference on the trained model, first select the model type trained in predict.py, then run:
   ```bash
   python predict.py
   ```

5. To visualize the embeddings, again, change the model name in plotting.py, then execute:
   ```bash
   python plotting.py
   ```
   This will generate a TSNE plotting of the model's clustering.

## Model Architectures
As state above, this project implements the following GNN architectures:
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE
- Simple Graph Convolution (SGC)

Each model has been encapsulated in the `modules.py` file, with a defined forward pass and relevant layers. The next sections contain brief descriptions of how each model functions.

### Graph Convolutional Networks (GCN)


### Graph Attention Networks (GAT)


### Graph Sample and Aggregation (GraphSAGE)


### Simplified Graph Convolution

