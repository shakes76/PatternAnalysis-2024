# Alzheimer's Disease Detection using Vision Transformer (ViT)
Yuvraj Fowdar, 47538209.

## Project Overview

This project focuses on using Vision Transformers (ViT) for the detection of Alzheimer's Disease from MRI brain scans, leveraging the ADNI dataset. The goal is to classify brain images into two categories:

- Alzheimer's Disease (AD)
- Healthy (No Alzheimer's)

The Vision Transformer model is utilized to process brain images and make predictions with a target test accuracy of 80%. This project includes dataset preprocessing, augmentation, and the implementation of a Vision Transformer architecture for image classification.

## Vision Transformer Architecture

## Repository Structure


- predict.py
Softmax applied onto model externally for probability predictions!!

## Preprocessing: Dataset Loading + Augmentation
- vision trasnfoemrs seem to want image sizes that are multiples of 16. So 224x224 might be the angle.
- Also need to normalise the data so we can help the model train better and converge faster.
- we tried multiple versiopns of data augmentation, settled down for this.
- 
Resize to 224x224.
<!-- Random Horizontal Flip.
Random Rotation (small angles, e.g., ±10 degrees).
Random Brightness/Contrast Adjustments (slight, e.g., ±20%). -->
Normalization using empirically gathered statistics from our ADNI dataset; basically found the normalisation from training dataset, applied this everywhere 
(only colelcted from train dataset so no data leakage occurs).

## Experiments

## Usage 
Run all files from the directory.

1. `pip install -r requirements.txt`
   
#### Training
... `terminal command to train stuff etc`, data path, plots, etc...

#### Testing
terminal command to test


## References

Ballal, A., 2023. Building a Vision Transformer from Scratch in PyTorch, Available at: https://www.akshaymakes.com/blogs/vision-transformer [Accessed 21 October 2024].

Dosovitskiy, A., et. al., 2020. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Available at: https://arxiv.org/pdf/2010.11929.pdf [Accessed 21 October 2024].

Learnpytorch.io, 2023. PyTorch Paper Replicating. Available at: https://www.learnpytorch.io/08_pytorch_paper_replicating/#8-putting-it-all-together-to-create-vit [Accessed 21 October 2024].


Lightning.ai, 2023. Vision Transformer (ViT) - PyTorch Lightning Tutorial. Available at: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html [Accessed 21 October 2024].
