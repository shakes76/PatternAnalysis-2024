# Lesion Detection using YOLOv8

## Problem Introduction
Lesion detection in skin images, such as those provided by the ISIC dataset, helps identify potential melanomas. The problem we aim to solve is automating the detection of lesions within the ISIC 2017/8 data set with a detection network
such as the YOLOv7 or newer with all detections having a minimum
Intersection Over Union of 0.8 on the test set and a suitable accuracy for classification.

## Dataset Introduction
![Examples from Dataset](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/Examples-of-images-belonging-to-the-ISIC-2017-dataset.png)

We used the **ISIC-2017 dataset**, which contains dermoscopic images labeled with ground-truth masks indicating skin lesions. The dataset provides images for training, validation, and testing purposes, with segmentation masks that serve as ground truth.

- **Training Set**: 2000 images  
- **Validation Set**: 150 images  
- **Test Set**: 1200 images  

Each image was preprocessed into bounding boxes, normalized for size consistency, and stored in YOLO-readable formats, which we will go more in depth later.

## Model: YOLOv8
YOLO (You Only Look Once) is a fast, accurate, and efficient object detection model. It’s suitable for real-time object detection applications. The model used in this project is the **YOLOv8** architecture, optimized for object detection with real-time performance. YOLOv8 allows for fine-tuning on custom datasets and provides the flexibility to balance speed and accuracy.

![YOLOv8 Architecture](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/yolo.png)

### Architecture
YOLOv8 consists of several stages involving **Convolutional Layers (Conv), Cross-Stage Partial Networks (C2f), Spatial Pyramid Pooling (SPPF), Upsampling, and Concatenation Operations**. Below are some intermediate visualizations captured during inference, illustrating how features evolve through different stages.

### Example Visualisations

<div align="center">

| **Stage 0 - Conv Features** | **Stage 9 - SPPF Features** |
|-----------------------------|-----------------------------|
| ![Stage 0 - Conv Features](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/stage0_Conv_features.png) | ![Stage 9 - SPPF Features](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/stage9_SPPF_features.png) |

| **Stage 13 - Upsample Features** | **Stage 21 - Final C2f Features** |
|----------------------------------|------------------------------------|
| ![Stage 13 - Upsample Features](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/stage13_Upsample_features.png) | ![Stage 21 - Final C2f Features](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/stage21_C2f_features.png) |

</div>

### How it works:
Convolutional Layers (Conv): These layers are responsible for learning local patterns and features such as edges and textures from the input image. The convolution operation helps the model reduce spatial dimensions while extracting meaningful representations.

Cross-Stage Partial Networks (C2f): The C2f blocks allow the model to improve gradient flow during backpropagation by splitting feature maps into two parts.One passes through the layers while the other skips them, enhancing information preservation and improving model learning.

Spatial Pyramid Pooling (SPPF): This component aggregates global context from different receptive fields, allowing the model to handle objects of varying sizes and shapes effectively.

Upsampling: This operation increases the resolution of feature maps enabling finer localisation of objects. It allows the network to combine low-level features with high-level features for better predictions.

Concatenation Operations: These operations merge feature maps from different stages allowing the model to combine both high-level and low-level information to make more precise predictions.