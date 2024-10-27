## Max Gadd
Student: 46985431
Email: mgadd02@gmail.com
Student Email: s4698543@uq.edu.au

### Elected Task:
Segment the HipMRI Study on Prostate Cancer using the processed 2D slices (2D images) with the 2D UNet with all labels having a minimum Dice similarity coefficient of 0.75 on the test set on the prostate label.

# 2D UNet Image Segmentation of HipMRI Prostate Cancer Study

This project implements a UNet-based model to perform semantic segmentation on the HipMRI Study dataset for prostate cancer. The goal is to accurately segment the prostate region from 2D MRI slices, achieving a minimum Dice Similarity Coefficient (DSC) of 0.75 on the test set for the prostate label.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [Data Loading](#data-loading)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Conclusion](#conclusion)
- [References](#references)

# Introduction

Semantic segmentation is a crucial task in medical imaging, enabling precise identification of anatomical structures and abnormalities. In this project, a UNet-based deep learning model is employed to segment the prostate gland from MRI images in the HipMRI Study dataset.

## Dataset

The HipMRI Study dataset consists of processed 2D MRI slices of the prostate region. The dataset is organized into training, validation, and test sets, each containing images and corresponding segmentation labels.

- **Training Set**:
  - Images: `./data/HipMRI_study_keras_slices_data/keras_slices_train`
  - Labels: `./data/HipMRI_study_keras_slices_data/keras_slices_seg_train`
- **Validation Set**:
  - Images: `./data/HipMRI_study_keras_slices_data/keras_slices_validate`
  - Labels: `./data/HipMRI_study_keras_slices_data/keras_slices_seg_validate`
- **Test Set**:
  - Images: `./data/HipMRI_study_keras_slices_data/keras_slices_test`
  - Labels: `./data/HipMRI_study_keras_slices_data/keras_slices_seg_test`

## Prerequisites

- Python 3.7 or higher
- PyTorch
- NumPy
- SciPy
- scikit-image
- nibabel
- matplotlib
- tqdm
- torchvision
*Note: Ensure that you have a compatible GPU and CUDA installed if you plan to train the model on a GPU.*

## Project Structure
```
│
├── data
│   └── HipMRI_study_keras_slices_data
|       |
│       ├── keras_slices_train
│       ├── keras_slices_seg_train
|       |
│       ├── keras_slices_validate
│       ├── keras_slices_seg_validate
|       |
│       ├── keras_slices_test
│       └── keras_slices_seg_test
│
├── modules.py
├── dataset.py
├── train.py
├── predict.py
│
├── __init__.py
├── development.ipynb
│
└── README.md
```

modules.py: Contains the simplified UNet model definition.

dataset.py: Custom dataset class for data loading and preprocessing.

train.py: Script to train the UNet model.

predict.py: Script to evaluate the model on the test set and compute the Dice coefficient.

requirements.txt: Lists all Python dependencies.

README.md: Project documentation.


# Implementation Details

## Data Loading
The SegmentationData class in dataset.py handles data loading, preprocessing, and augmentation. Key functionalities include:
Loading NIfTI Images: Uses nibabel to load MRI images and labels from NIfTI files.
Preprocessing:
- Resizing images to a consistent shape.
- Normalizing pixel intensity values.
- Converting labels to one-hot encoded format for multi-class segmentation.
- Data Augmentation (Optional): Applies random transformations such as flips and rotations to enhance model generalization.

## Model Architecture
A simple UNet architecture is implemented in modules.py:
Contracting Path:
- Consists of three downsampling steps with feature channels (64, 128, 256, 512).
- Each downsampling step includes a DoubleConv block and a max-pooling layer.
Expanding Path:
- Mirrors the contracting path with upsampling layers.
- Uses transposed convolutions for upsampling and concatenates with corresponding features from the contracting path.
Output Layer:
- A final 1x1 convolution reduces the number of channels to the number of segmentation classes.
Regularization Techniques:
- Dropout Layers: Introduced in DoubleConv blocks to prevent overfitting.
- Weight Decay: Applied in the optimizer to penalize large weights.
### About the UNET Model
The U-Net architecture is a convolutional neural network designed for 2D image segmentation tasks. It features a symmetrical encoder-decoder structure: the encoder path captures context by downsampling the input image through convolutional and pooling layers, while the decoder path reconstructs precise spatial details by upsampling. What sets U-Net apart are the skip connections that link corresponding layers of the encoder and decoder, allowing the model to combine coarse, high-level features with fine-grained, low-level details. This design enables the network to produce accurate segmentation maps that delineate objects within an image.
![2-Figure1-1](https://github.com/user-attachments/assets/bf69fb1c-3c98-42ea-a9d9-399f2e9846ec)
Diagram: Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation - Noori, M., Bahri, A., & Mohammadi, K. (2019).

### Training
The training script train.py performs the following steps:
- Data Loading: Loads training and validation datasets with optional augmentation.
- Model Initialization: Creates an instance of the simplified UNet model with the appropriate number of input channels and classes.
- Loss Function: Uses CrossEntropyLoss for multi-class segmentation.
- Optimizer and Scheduler: Uses the Adam optimizer with weight decay for regularization. Implements a learning rate scheduler to adjust the learning rate during training.
- Training Loop: Performs forward and backward passes. Updates model weights based on the computed gradients. Logs training and validation losses.

### Evaluation
The evaluation script predict.py:
- Loads the Trained Model: Loads the model weights from the latest checkpoint.
- Runs Inference: Performs segmentation on the test dataset.
- Computes Dice Coefficient: Calculates the Dice Similarity Coefficient for each class.
- Outputs Results: Prints the Dice scores, with an emphasis on the prostate label.

# Results
The model achieved a minimum Dice Similarity Coefficient of 0.75 on the test set for the prostate label. This indicates a high degree of overlap between the predicted segmentation and the ground truth, demonstrating the model's effectiveness in accurately segmenting the prostate region.

Final Dice Similarity Coefficient for each class on Unseen Test Set:
```
Class 0: Dice = 0.9968
Class 1: Dice = 0.9839
Class 2: Dice = 0.9194
Class 3: Dice = 0.9493
Class 4: Dice = 0.8975
Class 5: Dice = 0.8873
```

Predicted Segmentations compared to Ground Truth Labels
![Figure_1](https://github.com/user-attachments/assets/5bf47e39-b6ba-4f52-b104-ad6224bdf554)
![Figure_2](https://github.com/user-attachments/assets/53090a01-6230-4f63-8892-ef3c0fac992f)
![Figure_3](https://github.com/user-attachments/assets/ef4f912b-0866-4328-a24b-7050cd381f4f)
![Figure_4](https://github.com/user-attachments/assets/94536202-f570-411f-9e4f-267cb6e8d70e)
![Figure_5](https://github.com/user-attachments/assets/e30332b6-cfac-49de-8b87-311909880639)


Dice Coefficients Per Training Epoch
![Dice_Epochs](https://github.com/user-attachments/assets/19052a9e-ceb5-46d0-9cd6-8d2f88f4e2bd)


# Usage

## Training the model
1. Prepare the Data: Ensure that the dataset is organized as specified in the Dataset section.

2. Run the Training Script:
```python train.py```
3. Monitor Training: Training progress, including loss values, will be displayed in the console. Model checkpoints will be saved after each epoch.

## Evaluating the model
1. Ensure Model Checkpoints are Available: The predict.py script will load the latest model checkpoint from the current directory.
2. Run the Evaluation Script:
```python predict.py```
3. View the Results: The script will output the Dice Similarity Coefficient for each class.

# Conclusion
This project demonstrates the application of a simplified UNet model for semantic segmentation of the prostate gland in MRI images. By addressing overfitting through model simplification, regularization, and data augmentation, the model achieves a high Dice Similarity Coefficient, indicating accurate segmentation performance.

# References
- UNet Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/2109.05443)
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Nibabel Documentation: https://nipy.org/nibabel/
- HipMRI Study Dataset: https://data.csiro.au/collection/csiro:51392v2?redirected=true.
- UNET diagram from paper on Brain Segmentation: Noori, M., Bahri, A., & Mohammadi, K. (2019). Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation. 2019 9th International Conference on Computer and Knowledge Engineering (ICCKE), 269-275.
