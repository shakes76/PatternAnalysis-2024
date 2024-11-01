# U-Net Image Segmentation

## Overview
This project implements a U-Net architecture for MRI segmentation tasks, mainly on Hip MRIs. The U-Net model is made to learn the features to provide accurate segmentations. This project has data preproccessing, training, predictions and evaluation metrics included.

## Table of Contents
- [Features](#features)
- [Usage](#usage)
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

## Installation
Clone the repository and install the required dependencies

## Hyperparameters
- Learning Rate: 1e-5
- Epochs = 5
- Batch size = 4

## Training and Evaluation
- Average Training Loss: ~0.001
- Average Evaluation Loss: ~0.04

## Results
- Average Dice Score: TBC
