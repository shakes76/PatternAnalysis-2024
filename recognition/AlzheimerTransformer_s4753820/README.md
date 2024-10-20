# Alzheimer's Disease Detection using Vision Transformer (ViT)
Yuvraj Fowdar, 47538209.

## Project Overview

We will attempt to use the latest vision transformers in order to classify Alzheimers from ADNI brain image data.
This project implements a Vision Transformer (ViT) model for the task of detecting Alzheimer's Disease using MRI brain scans. The goal of the model is to classify brain images as either having Alzheimer's disease or being healthy (no Alzheimer's) with transformers. Aiming for a 0.8 test accuracy is ideal.



## Repository Structure

#### predict.py
Softmax applied onto model externally for probability predictions!!

## Preprocessing: Dataset Loading + Augmentation
- vision trasnfoemrs seem to want image sizes that are multiples of 16. So 224x224 might be the angle.
- Also need to normalise the data so we can help the model train better and converge faster.
- 
Resize to 224x224.
<!-- Random Horizontal Flip.
Random Rotation (small angles, e.g., ±10 degrees).
Random Brightness/Contrast Adjustments (slight, e.g., ±20%). -->
Normalization using the standard pre-trained stats for ImageNet  / empirically gathered statistics.


### Run files from this directory!