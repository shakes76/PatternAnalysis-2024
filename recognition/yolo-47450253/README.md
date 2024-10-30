# Detecting Skin Lesions using YOLO11
Since 2016, the International Skin Imaging Collaboration (ISIC) have run annual challenges with the aim of developing imaging tools to aid in the automated diagnosis of melanoma from dermoscopic images. This repository focuses on detecting skin lesions within dermoscopic images. This data could then be taken and used in furthor diagnostic tools and is a useful first step in achieving this. As specified by the task requirements, this repository makes use of YOLOv11 to perform image detection on the provided dataset by the ISIC.

## About YOLO11
YOLOv11 is the 11th major iteration in the "You Only Look Once" family of object detection models and is a very recent development, having released in September 2024. YOLOv11 provides significant increases in accuracy and small object detection in comparison to previous versions, while still maintaining high efficiency and speed. The YOLO family of models are very flexible and can easily be trained to detect objects on a variety of custom datasets with little to occasionally no fine tuning needed provided that the dataset and labels are provided to it in the necessary formats.

### Model Architecture
YOLOv11 provides significant advancements over older versions of YOLO, with various improvements to the components that make up its architecture, but ultimately follows a very similar structure to its predecessors.

![Architecture Diagram of YOLOv11](/images/YOLOv11Architecture.png)
*Correction: the SPFF block in this diagram should be refered to as SPPF*

The Architecture can be broken down into a backbone (the primary feature extractor), neck (Intermediate Processing) and head (Prediction), stages of each segment being comprised of the following blocks:

#### Convolutions (Conv)
A basic convolution operation consisting of a Conv2d, BatchNorm2d, and SiLU activation function

#### Cross Stage Partial - with kernal size of 2 (C3K2)
A computationally efficient block that can switch between a more standard implementation of CSP or a mode which can extract more complex features. This block is used frequently throughout the model's architecture to aggregate, process and refine features. This block splits feature maps in two and only running one part through its layers, while the other skips through and is then concatenated.

#### Spacial Pyramid Pooling - Fast (SPPF)
This component allows the model to divide the image into a grid and then pool the features of each cell indepentently, allowing the model to work on different image resolutions. SPPF is a faster version of typical Spacial Pyramid Pooling that trades accuracy for speed.

#### Attention Mechanism (C2PSA)
The biggest change between YOLOv11 and its last major predecessor YOLOv8, this block allows the model to focus on important portions of the image, improving detection of small or obscured objects. This likely provides significant improvements for this task specifically due to potential occlusion of skin lesions by body hair.

### Visualisations By Block Type
|   |   |
|---|---|
| Start: Stage 0 - Conv Features | Pooling: Stage 9 - SPPF Features|
|![Conv Features](/images/stage0_Conv_features.png)|![SPPF Features](/images/stage9_SPPF_features.png)|
|Attention Mechanism: Stage 10 - C2PSA Features| Final: Stage 22 - C3K2 Features|
|![C2PSA Features](/images/stage10_C2PSA_features.png)|![C3K2 Features](/images/stage22_C3k2_features.png)|

## About the ISIC2018 Dataset
### Dataset Breakdown
The [ISIC 2018 Task 1](https://challenge.isic-archive.com/data/#2018) dataset is comprised of 3694 dermoscopic full colour images broken down into three categories:
- 2594 Training Images
- 100 Validation Images
- 1000 Testing Images

Each of these categories is also accompanied by the same number of black and white Ground Truth masks.

|Dermoscopic Image|Ground Truth|
|---|---|
|![Dermoscopic Example](/images/ISIC_0000000.jpg)|![Ground Truth Example](/images/ISIC_0000000_segmentation.png)|



## Requirements (Polish with exact versions later)
torch
torchvision
ultralytics
opencv

## Dataset File Structure
The dataset should be stored in the following file structure
```
data
|- train
    |- labels
    |- images
|- validate
    |- labels
    |- images
|- test
    |- labels
    |- images
```

## References
https://medium.com/@nikhil-rao-20/yolov11-explained-next-level-object-detection-with-enhanced-speed-and-accuracy-2dbe2d376f71
https://arxiv.org/html/2410.17725v1
https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5