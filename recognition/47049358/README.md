# Improved 3D Improved UNet

## The Model

<p align="center">
  <img src = documentation/model_architecture.png alt = "Improved 3D UNet Architecture" width = 80% >
<p>
  
 UNet is an architecture for convolutional neural networks specifically for segmentation tasks. The model takes advantage of skip connections and tensor concatenations to preserve input details and its structure while learning appropriate segmentations. 

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