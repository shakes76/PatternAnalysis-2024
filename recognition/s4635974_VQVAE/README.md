# VQ-VAE generative model on Hip MRI Study for Prostate Cancer Dataset.

## Project Aim
The aim of this project was to create a generative Vector Quantized - Variational Austoencoder (VQ-VAE) model using the Hip MRI Prostate Cancer 2D dataset [1]. The model must produce "reasonably clear images" and have a Structural Similarity Index Measure (SSIM) of over 0.6.

## Model Description

### Overview
The VQ-VAE is an unsupervised generative model that encodes an image into a discrete latent representation and then decodes the discrete representation back into a high-quality representation of the orignal image. 

The distinguishing feature of the VQ-VAE is that the latent representation is discrete, rather than continuous. This is achieved by using vector quantisation (VQ). 

The model builds on the Vareational Auto Encoder (VAE), which encodes images into a continous latent representation. This model suffered from "posterier collapse", where the latent space is ignored when paired with a powerful autoregressive model. By leveraging vector quantisation, the VQ-VAE is able to avoid this problem. 

- Overview
- Problem it solves

### Model Architecture Overview
- Include picture

### Loss Function

### How it works

The discrete latent representation is a compressed lower-dimensional version of the original data that captures essential features. It consists of a finite set of codebook vectors, also known as "embeddings" and are updated via the models performace during training.

### Specific Architecture
- pytorchinfo 

### Hyperparameters

### Data Pre-processing
- Describe any specific pre-processing you have used with references if any.
- Justify your training, validation and testing splits of the data.


## Model Results

### Training

### Testing

## Usage
    - list any dependencies required
    - versions
    - how to reproduce results

### Environment setup

### Dataset Access
- Include file format

### Running the Model



## References

1. Early stopping algo
2.
3.









