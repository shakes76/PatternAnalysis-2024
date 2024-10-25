# YOLOv8 for Skin Lesion Detection: ISIC 2018 Challenge - Thanh Trung Pham (46717481)

## Problem and algorithm

This repository makes use of YOLOv8, a highly accurate and fast object detection model (but modified to do segmentation task) in order to tackle the ISIC 2018 Skin Lesion Challenge, it is a challenge focused on the dagnosis of dangerous skin issues such as melanoma. This work leverages the strength of YOLO algorithm in order to segment skin lesions. 
YOLO (You only look once) algorithm for object detection works by first dividing the input images into a S x S grid of cells, then, each cell's responsibility is detecting objects whose center is within that cell, by doing this YOLO is superior to many other object detection algorithms such as R-CNN as it is able to do works simultaneously and 'only look once' through the entire images, making its speed ideal for real-time applications. 
Each cell in the grid needs to predict many bounding boxes and generate a confidence score for all of the boxes, where each bounding box's defined by using: 'x' representing x-coordinate of the center of the bounding box with respect to the cell, 'y' representing y-coordinate of the center of the bounding box with respect to the cell, and 'w' and 'h' representing width and height of the box with respect to the image itself. Moreover, a confidence score is also generated for each cell, it is a product of the probability of the object being there and the IoU (Intersection over union) between the box and the ground truth. Each cell can produce many bounding boxes but then Non-Maximum Suppression (NMS) is applied to prevent detection of multiple same objects hence getting rid of redundant ones, this is done by sorting the bounding boxes  -> select the box with the highest confidence score -> removing boxes that have high IoU score with this box -> repeat until boxes do not overlap too much anymore (no more redundant boxes). 
YOLO also makes use of CNN network as its backbone for feature extraction. It also leverages 'Feature fusion' where features at different scales when being extracted are combined to 'see the bigger picture'. YOLO algorithm loss is a combination of three losses: The first one is bounding box loss to make sure predicted boxes are close to ground truth boxes, second one is 'object presence/objectness' loss designed to penalizes false positives where YOLO detect an object (of any class) when there's only the background (it measures the confidence of the model when it comes to the presence of objects in a bounding box), the third is the classification loss to ensure the predicted probabilities for classes match the ground truth.
YOLOv8-segmentation goes a step further and instead of just identifying the bounding boxes, it segments each individual object's pixel (assigning pixel-wise labels to objects in images). YOLOv8-segmentation includes another branch in its architecture to detect segmentation masks for all detected instances by bounding boxes.

```
YOLOv8 segmentation:
├── Backbone: CSPDarknet
├── Neck: PANet
└── Head: Decoupled Detection Head
    ├── Classification Branch
    └── Regression Branch
```

![Alt text](figures/figure1.png?raw=true "YOLOv8 architecture")
## Requirements

```
python>=3.8
ultralytics==8.0.58
opencv-python>=4.1.2
albumentations==1.4
scikit-learn
scikit-build
```
# Preprocessing

The dataset provided has ground truth labels being a binary mask, however, YOLOv8-segmentation from ultralytics accepts a different type of label, that is a polyglon, therefore Opencv was utilised to generate contours from the mask in order to create a polyglon that can be used to train.

The train-validation/test split is 80-20.

# Results
![Alt text](figures/results.png?raw=true "Training results")

It is observed that the loss for the algorithm decreases steadily over 30 epochs. 
The metric mAP50-95(M) is mean average precision where a 'true' positive is defined by having Intersection over Union (IoU) with the ground truth mask from 0.5-0.95, the algorithm is able to reach 0.7318 mAP50-95 on the test set after 30 epochs. 

These are some detections predicted by the fine-tuned YOLOv8-segmentation visualized.
![Alt text](figures/prediction_test.jpg?raw=true "Sample prediction 1")
![Alt text](figures/prediction_test2.jpg?raw=true "Sample prediction 2")

## Acknowledgments

- ISIC 2018 Challenge organizers
- Ultralytics for YOLOv8
- Contributors to the ISIC archive
