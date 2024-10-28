# 3D UNet for Prostate Segmentation

## Introduction

This project utilizes the 3D UNet architecture to train on the Prostate 3D dataset, aiming to achieve precise medical volumetric image segmentation. We evaluate the performance of the segmentation using the Dice similarity coefficient, targeting a minimum score of 0.7 for all labels on the test set. Image segmentation transforms a volumetric image into segmented areas represented by masks, which facilitates medical condition analysis, symptom prediction, and treatment planning.

## Background

### UNet-3D

The 3D UNet is an extension of the original UNet architecture, which is widely used for segmenting 2D medical images. While the standard UNet processes 2D images, UNet-3D extends this functionality to volumetric (3D) images, allowing for more accurate segmentation of complex medical structures found in modalities like MRI or CT scans.

UNet architecture leverages a combination of convolutional neural networks (CNNs) and skip connections, improving performance by combining high-resolution features from the contracting path with low-resolution context from the expansive path. This design maintains spatial information throughout the segmentation process, which is critical in the medical imaging field.


![3D U-Net Architecture](https://raw.githubusercontent.com/Han1zen/PatternAnalysis-2024/refs/heads/topic-recognition/recognition/3D-UNT%2048790835/picture/3D%20U-Net.webp)

### Dataset

For this project, we will segment the downsampled Prostate 3D dataset. A sample code for loading and processing Nifti file formats is provided in Appendix B. Furthermore, we encourage the use of data augmentation libraries for TensorFlow (TF) or the appropriate transformations in PyTorch to enhance the robustness of the model.

### Evaluation Metric

We will employ the Dice similarity coefficient as our primary evaluation metric. The Dice coefficient measures the overlap between the predicted segmentation and the ground truth, mathematically expressed as:

\[ \text{Dice} = \frac{2 |A \cap B|}{|A| + |B|} \]

where \( A \) and \( B \) are the sets of predicted and ground truth regions respectively. A Dice coefficient of 0.7 or greater indicates a significant degree of accuracy in segmentation.

## Objectives

- Implement the 3D Improved UNet architecture for the Prostate dataset.
- Achieve a minimum Dice similarity coefficient of 0.7 for all labels on the test set.
- Utilize data augmentation techniques to improve model generalization.
- Load and preprocess Nifti file formats for volumetric data analysis.

## Quick Start

To get started with the 3D UNet model for prostate segmentation, follow these steps:

1. **Clone the Repository**: Clone the repository to your local machine.
2. **Install Dependencies**: Ensure you have the required libraries installed.
3. **Prepare the Dataset**: Download the Prostate 3D dataset and place it in the `data/` directory.
4. **Run Training**: Execute the training script to begin training the model on the Prostate 3D dataset.

## Results

### Training and Validation Loss

![Training and Validation Loss](https://github.com/Han1zen/PatternAnalysis-2024/blob/topic-recognition/recognition/3D-UNT%2048790835/picture/train_loss_and_valid_loss.png#:~:text=loss.jpg-,train_loss_and_valid_loss,-.png)

- The **training loss** curve demonstrates a rapid decline in the early stages of training, indicating that the model is effectively learning and adapting to the training data.
- As training progresses, the loss stabilizes, ultimately reaching around **0.6**. This suggests that the model performs well on the training set and is capable of effective feature learning.

- The **validation loss** curve also exhibits a downward trend, remaining relatively close to the training loss in the later stages of training.
- This indicates that the model has good generalization capabilities on the validation set, with no significant signs of overfitting. The validation loss stabilizes at approximately **0.62**, further supporting the model's effectiveness.

### Dice Similarity Coefficient

![Dice](https://github.com/Han1zen/PatternAnalysis-2024/blob/topic-recognition/recognition/3D-UNT%2048790835/picture/dice.png#:~:text=dice.-,png,-loss.jpg)
- The model achieves a **Dice similarity coefficient** of over **0.7** for all labels, meeting our established target.
- This indicates that the model performs excellently in the segmentation task, accurately identifying and segmenting different regions of the prostate.


## References

1. Sik-Ho Tsang. "Review: 3D U-Net — Volumetric Segmentation (Medical Image Segmentation)." [Towards Data Science](https://towardsdatascience.com/review-3d-u-net-volumetric-segmentation-medical-image-segmentation-8b592560fac1).

