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
