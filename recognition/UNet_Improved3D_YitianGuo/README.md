
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
- Python 3.11
- PyTorch 2.0.1
- Monai 1.2.0
- Numpy 1.26.4

```
pip install -r Dependencies.txt
```