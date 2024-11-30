# U-Net Image Segmentation

## Overview
This project implements a U-Net architecture for MRI segmentation tasks, mainly on Hip MRIs. The U-Net model consist of two main parts, an encoder and decoder, the encoder progressibely reduces the spatial dimensions of the input image while extracting features, while the decoder reconstructs the image by upsampling and refining features. The model uses skip connections between the encoder and decoder to retain spatial information. A bottleneck layer sits between the encoder and decoder, connecting the two.
This project has data preproccessing, training, predictions and evaluation metrics included.

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Hyperparameters](#hyperparameters)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Features
- U-Net architecture
- Design to effectively segment MRI images of Nifti format
- Overfitting measures with early stopping 
- Hyperparameters are adjustable
- Evaluation metrics including Dice Score and Loss
- CUDA ready with Cpu backup
- Predictions on new unseen images

## Dependencies
- Pytorch 2.3.0
- Numpy 1.26.4
- Nibabel 
- Albumentations 1.2.0
- tqdm 4.66.5

## Hyperparameters
- Learning Rate: 1e-5
- Epochs = 5
- Batch size = 4

## Training and Evaluation
- Average Training Loss: ~0.001
- Average Evaluation Loss: ~0.04

## Results
- Average Dice Score: TBC

