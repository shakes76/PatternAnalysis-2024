# Alzheimer's Disease Classification Using Vision Transformers (Task 5)

Author: John Kong 4697959

## Problem Description

This project focuses on classifying Alzheimer's disease (AD) and normal cognition (NC) using brain MRI images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. We use a Vision Transformer (ViT) model to achieve this classification, leveraging its powerful capabilities in image recognition tasks. The goal is to distinguish between AD and NC with high accuracy, providing a deep learning-based approach to support medical diagnosis.

## The Model and How it works

The Vision Transformer (ViT) applies transformer architecture, which has been highly successful in natural language processing, to image data. Instead of relying on convolutions, the ViT divides each image into smaller patches, treats them as sequences, and processes them through multi-head self-attention mechanisms. The model learns global relationships between image patches, allowing it to capture complex spatial patterns that are critical for accurate classification. [1]

## Model Architecture

Vision Transformer Architecture:

![ViT model](Readme-imgs/VIT_architecture.png)

The Vision Transformer consists of several key components [2]:

1. **Patch Embedding**:

The input image is divided into fixed-size patches (e.g., 16x16 pixels). Each patch is flattened into a vector and linearly transformed into an embedding. These patch embeddings serve as the input tokens to the transformer.

2. **Positional Encoding**:

Since transformers do not inherently understand the spatial relationships between patches, positional encodings are added to the patch embeddings. These encodings provide information about the position of each patch within the image, enabling the model to consider spatial order.

3. **Transformer Encoder**:

The encoder consists of multiple layers of multi-head self-attention and feed-forward neural networks. The attention mechanism allows the model to focus on different patches when making predictions, while the feed-forward layers capture complex patterns and relationships between patches.

4. **Classification Head**:

A special classification token `CLS` is added to the sequence of patch embeddings. This token aggregates information from all patches through the transformer layers. The final state of this token is used as the image representation, which is passed through a fully connected layer to produce the output class (AD or NC).

## Implementation Details in `modules.py`

The code for the Vision Transformer is organized into four main classes [3]: 

1. **`PatchEmbedding`**:

This class splits the input image into patches and embeds each patch into a fixed-size vector using a linear layer. The patch embeddings are then combined with positional encodings to retain spatial information.

2. **`MultiHeadSelfAttention`**:

This class implements the multi-headed self-attention mechanism. The multi-head self-attention mechanism computes attention scores for each patch, determining the importance of different patches relative to each other. This mechanism is implemented using matrix multiplication and scaled dot-product attention. 

3. **`TransformerEncoderBlock`**: 

This class defines the structure of each transformer block, consisting of a multi-headed self-attention layer followed by a feed-forward neural network with skip connections and layer normalization. The feed-forward layer consists of two linear layers with a GELU (Gaussian Error Linear Unit) activation in between. The feed-forward network is applied independently to each token.

4. **`VisionTransformer`**:

This is the main class for the Vision Transformer model. It integrates all the components mentioned above and defines the forward pass for the entire architecture, from patch embedding to the classification head. 

## Implementation Details in `dataset.py`

This file is responsible for data loading and preprocessing. It includes the following components [4]:

1. **Data Loading**:

The file defines functions to load the ADNI dataset and split it into training, validation, and test sets. The dataset is split with an 80-20 rule for training and validation, while a separate set is used for testing. The `get_data_loaders` function uses PyTorch’s `DataLoader` class to load and batch the data efficiently.

2. **Transforms**:

Several image transformations are applied to preprocess the data, including resizing, normalization, and augmentation such as random flips or rotations. Specifically, the dataset is normalized using `ImageNet`'s means and standard deviations and the images are also resized to `224x224` pixels, a typical input size for Vision Transformers [5]. These augmentations are designed to increase data diversity, helping the model generalize better to unseen images.
 
3. **Data Pipeline**:

The dataset is split into training (80%) and validation (20%) sets. An independent test set is used to evaluate the final model’s performance. The function `get_data_loaders` returns PyTorch DataLoader objects for training, validation, and testing. It accepts parameters like batch size and validation split ratio.

## Data Configuration and Data Example
 
The dataset is organized in the following folder structure and can be obtain from the Rangpur Path: `/home/groups/comp3710/ADNI`.

```
AD_NC
├── test
│   ├── AD
│   └── NC
└── train
    ├── AD
    └── NC
```

Below are some example of input images of the different classes:

AD:

![AD image 1](Readme-imgs/sample_train/AD/388206_85.jpeg)
![AD image 2](Readme-imgs/sample_train/AD/388206_86.jpeg)

NC:

![NC image 1](Readme-imgs/sample_train/NC/1182968_105.jpeg)
![NC image 2](Readme-imgs/sample_train/NC/1182968_106.jpeg)

## Dependencies

To run this project, you will need the following dependencies:

```
Python 3.8+
PyTorch 1.12+
torchvision 0.13+
scikit-learn 1.0+
matplotlib 3.5+
numpy 1.21+
```
