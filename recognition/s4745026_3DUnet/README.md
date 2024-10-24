# 3D Unet for Prostate MRI Segmentation

## Overview

This project implements a Basic 3D Unet model that segments the Prostate MRI dataset. The purpose is to segment the downsampled dataset with a minumum Dice Score of 0.7 when testing. The algorithm identifies and segments the different regions of the scans to an accurate degree.

The dataset used contains 3D images of the prostate, which has been downsampled to a manageable size.

## Algorithm Description

The 3D UNet model is a full convolutional neural network which has been designed for 3D image segmentation. It uses the classic encoder-decoder architecture with the bottleneck.

- Encoder: The encoder captures features from the input and reduces the spatial dimensions to capture high-level features.
- Bottleneck: The bottleneck layer connects the encoder to the decoder which is the most dense layer of the input data.
- Decoder: The decoder upsamples the features back into the original spatial dimension while integrating the features from the encoder using skip connections.

The model is trained to maximise the Dice Similarity coefficient on the training data which is used to get accurate segmentation of the prostate. Data transformations are applied to improve the robustness and prevent overfitting, these include rotations, flips and resizing.

## Dependencies

To run this project, you will need the following dependencies:

| Library      | Version |
| ------------ | ------- |
| Python       | 3.8+    |
| PyTorch      | 1.10+   |
| Torchvision  | 0.11+   |
| Nibabel      | 3.2+    |
| NumPy        | 1.19+   |
| Matplotlib   | 3.4+    |
| Scikit-learn | 0.24+   |

All dependencies are required to reproduce the results.
