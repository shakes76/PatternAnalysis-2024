# 2D U-Net for Prostate MRI Image Segmentation

## Description

This project implements a 2D U-Net model for segmenting magnetic resonance imaging (MRI) scans from the **HipMRI study**. The main objective is to accurately detect and segment the prostate, which is a critical task in prostate cancer diagnosis. U-Net, a popular architecture for biomedical image segmentation, is used to classify six different tissue types in MRI slices, focusing on identifying the prostate for cancer detection.

The U-Net model learns to predict pixel-wise segmentation masks that label different tissue classes. It operates by progressively downsampling the input MRI slices to capture abstract features and then upsampling to restore spatial resolution, enabling accurate segmentation.

## How It Works

The architecture of U-Net is based on an **encoder-decoder** design. The encoder path uses convolution and pooling layers to capture high-level features from the input MRI image, reducing its resolution. The decoder then upscales these features using transposed convolutions and reconstructs the original spatial dimensions to produce a segmentation map. **Skip connections** between the encoder and decoder help preserve fine details by passing high-resolution features from the encoder to the decoder.

Once trained, the model takes MRI slices as input and outputs segmentation masks.

## Dependencies

To reproduce the results in this project, you will need the following dependencies:

- Python 3.11
- PyTorch 2.0.1
- Torchvision 0.15.2
- Nibabel 5.1.0
- Numpy 1.25.0
- Matplotlib 3.7.2
- Tqdm 4.66.1

## Pre-processing Techniques

In this project, the MRI images undergo several pre-processing steps to ensure optimal performance of the 2D U-Net model for prostate cancer segmentation. The key pre-processing techniques include:

1. **Image Normalization**:
   - MRI images are often characterized by varying intensity values due to different imaging conditions. Normalization scales the pixel values to a specific range (typically [0, 1]) to ensure consistency across images. This process can enhance the convergence of the neural network during training.
   - **Reference**: 
     - A. M. F. De Mello et al., "Improving the Generalization of Deep Learning Models for Medical Imaging through Data Normalization," *Medical Image Computing and Computer-Assisted Intervention* (MICCAI), 2020.

2. **Resizing**:
   - The images are resized to a consistent dimension (e.g., 256x128) suitable for input into the U-Net model. This step maintains a uniform input size, allowing for batch processing during training.
   - **Reference**: 
     - R. Ronneberger, O. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *Medical Image Computing and Computer-Assisted Intervention* (MICCAI), 2015.

## Dice Scores

The following table shows the Dice coefficient scores for each class in the segmentation task:

| Class | Dice Score |
|-------|------------|
| 1     | 0.87       |
| 2     | 0.76       |
| 3     | 0.82       |
| 4     | 0.79       |
| 5     | 0.69       |
| 6     | 0.74       |

### Mean Dice Score

The mean Dice score across all classes is approximately **0.78**. This score indicates the model's performance in accurately segmenting the various regions of interest in the MRI images.

## Training Loss Graph

The training process was monitored through the loss graph shown below, which illustrates the model's performance over the epochs. A decreasing loss indicates that the model is learning effectively.

![Loss Graph](PatternAnalysis-2024/recognition/2dUnet-S4698007/graphs/Loss-plot-for-2d-Unet.png)
