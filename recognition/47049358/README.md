---
title: COMP3710 Report
author: "Ryuto Hisamoto"
date: "2024-10-24"
bibliography: references.bib
---

# Improved 3D Improved UNet

## The Model

<p align="center">
  <img src = documentation/model_architecture.png alt = "Improved 3D UNet Architecture" width = 80% >
<p>

- Negative Slop: $10 ^ {-2}$
- Dropout Probability: 0.3

  
 UNet is an architecture for convolutional neural networks specifically for segmentation tasks. The model takes advantage of skip connections and tensor concatenations to preserve input details and its structure while learning appropriate segmentations. The basic structure of UNet involves the downsampling and upsampling of original images with skip connections in between corresponding pair of downsampling and upsampling layers. Skip connection is a technique used to (1) preserve features of the image and (2) prevent diminishing gradients over deep layers of network. On top of it, the improvement on UNet proposed by authors of "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge" [@IsenseeFabian2018BTSa]involves the integration of segmentation layers at different levels of the architecture.  
 

 |Labels|Segment|
| --------------- | ---------------------------------------- |
|0| Background |
|1| Body |
|2| Bones |
|3| Bladder |
|4| Rectum |
|5| Prostate |

### Loading Data

The authors of "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge" seem to have used the following augmentation methods:

- Random rotation
- Random scaling
- Elastic transformation
- Gamma correction
- Mirroring (assumably horizontal flip given their problem space)

However, some augmentations methods are altered to limit the complexity of solution. For instance, use of elastic transformation was avoided as it could alter the image significantly, causing it to deviate from the actual images the model may find. Moreover, the tuning of such complex method could decrease the maintainability of solutin. Therefore, the project preserved basic augmentation techniques to process the training data. More precisely, techniques used are limited to:

- Random Rotation ($[-0.5, 0.5]$ for all x, y, and z coordinates)
- Random Vertical Flip
- Gaussian Noise ($\mu = 0, \sigma = 0.5$)

In addition, all images are normalised as they are loaded to eliminate difference in intensity scales if there are any. Moreover, all voxel values are loaded as `torch.float32` but `torch.uint8` is used for labels to save memory consumption.

### Training

- Batch Size: 2
- Number of Epochs: 300
- Learning Rate: $5e ^ {-4}$
- Initial Learning Rate (for lr_scheduler): 0.985
- Weight Decay: $1e ^ {-5}$

The model takes in an raw image as its input, and its goal is to learn the best feature map which ends up being a multi-channel segmentation of the original image. 

### Loss Function

The model utilises dice loss as its loss function. Moreover, it is capable of using deviations of dice loss such as a sum of dice loss and cross-entropy loss, or focal loss. A vanilla dice loss has formula: $$D(y_{true}, y_{pred}) = 2 \times \frac{\Sigma(y_{true} \cdot y_{pred})}{\Sigma y_{true} + \Sigma y_{pred}}$$

in which $y_{true}$ is the ground truth probability and $y_{pred}$ is the predicted probability. The loss function mitigates the problem with other loss functions such as a cross-entropy loss which tend to be biased toward a dominant class. The design of dice loss provides more accurate representation of the model's performance in segmentation. In addition `monai` provides an option to exclude background from the calculation of loss, and the model makes use of this option when calculating the loss (background is included when testing).

### Optimiser

**Adam (Adaptive Moment Estimation)** is an optimisation algorithm that boosts the speed of convergence of gradient descent. The optimiser utilises an exponential average of gradients, which allows its efficient and fast pace of convergence. Moreover, the optimiser applies a $L_2$ regularisation to penalise for the complexity of model.
In addition, the model utilises a learning rate scheduler based on the number of epochs, which dynamically changes the learning rate over epochs. This allows the model to start from a large learning rate which evntually settles to a small learning rate for easier convergence.
It is to be noted that mixed precision and gradient accumulation are used to reduce the memory consumption during the training.

 ### Testing

The model is tested by measuring its dice scores on the segmentations it produces for unseen images. Although the model outputs softmax values for its predicted segmentations, they are one-hot encoded during the test to maximise the contribution of correct predictions. Moreover, unlike tranin

 ## The Problem

 Segmentation is a task that requires a machine learning models to divide image components into meaningful parts. In other words, the model is required to classify components of an image correctly into given labels.

 

(Dice Coefficient)[https://en.wikipedia.org/wiki/Dice-Sørensen_coefficient]

(Loss Function)[https://www.sciencedirect.com/science/article/pii/S1746809422007030]

