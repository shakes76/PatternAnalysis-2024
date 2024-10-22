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
