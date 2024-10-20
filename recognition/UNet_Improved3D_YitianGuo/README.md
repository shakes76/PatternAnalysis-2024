
# 3D Improved UNet for Prostate Segmentation

## Task description:
This project implements a 3D Improved UNet model to segment prostate regions from downsampled MRI scans. 
The goal is to achieve a minimum Dice similarity coefficient of 0.7 on the test set for all labels.

## Problem Statement


## Model Description

## Project Structure
- `modules.py`: Contains the implementation of the 3D Improved UNet model.
- `dataset.py`: Loads and preprocesses the downsampled prostate MRI dataset.
- `train.py`: Scripts for training, validation, testing, and saving the model.
- `predict.py`: Provides inference examples and visualization of segmented regions.
- `README.md`: This file, documenting the project.

## Dependencies
- Python==3.11
- torch==2.0.1
- monai==1.3.2
- numpy==1.26.4
- matplotlib==3.9.2
- nibabel==5.3.0
- scikit-learn==1.5.1

Use the following command to create the conda environment
```
conda env create -f environment.yml
```
activate environment
```
conda activate pytorch-2.0.1
```