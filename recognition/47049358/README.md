# Improved 3D Improved UNet

## The Model

<p align="center">
  <img src = documentation/model_architecture.png alt = "Improved 3D UNet Architecture" width = 80% >
<p>
  
 UNet is an architecture for convolutional neural networks specifically for segmentation tasks. The model takes advantage of skip connections and tensor concatenations to preserve input details and its structure while learning appropriate segmentations. The basic structure of UNet involves the downsampling and upsampling of original images with skip connections in between corresponding pair of downsampling and upsampling layers. Skip connection is a technique used to (1) preserve features of the image and (2) prevent diminishing gradients over deep layers of network.\
 

 |Labels|Segment|
| --------------- | ---------------------------------------- |
|0| Background |
|1| Body |
|2| Bones |
|3| Bladder |
|4| Rectum |
|5| Prostate |

 ### Training

The model takes in an raw image as its input, and its goal is to learn the best feature map which ends up being a smulti-channel segmentation of the original image.

 ### Testing



 ## The Problem

 Segmentation is a task that requires a machine learning models to divide image components into meaningful parts. In other words, the model is required to classify components of an image correctly into given labels.

 

(Dice Coefficient)[https://en.wikipedia.org/wiki/Dice-Sørensen_coefficient]

Updates:

**Data Augmentation**

- Random Rotation
- Random Scaling
- Normalisation
- Gamma Correction
- Vertical Flip

(Loss Function)[https://www.sciencedirect.com/science/article/pii/S1746809422007030]