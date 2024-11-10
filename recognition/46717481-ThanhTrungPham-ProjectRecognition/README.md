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

Model Training Observations
The training process of the fine-tuned YOLOv8 model was monitored over 30 epochs, during which a steady decline in the loss function was observed. This indicates that the model effectively minimized the error and adapted well to the underlying patterns in the training data. The decreasing trend of the loss function suggests a successful learning process, as the model adjusted its weights to better fit the provided examples.

Evaluation Metric: mAP50-95 (Mean Average Precision)
The key evaluation metric used to assess the model's performance is the mean Average Precision (mAP50-95). This metric measures the model's ability to correctly identify and segment objects across a range of Intersection over Union (IoU) thresholds, from 0.5 to 0.95 with a step size of 0.05. The IoU threshold determines the extent of overlap required between the predicted mask and the ground truth mask for a prediction to be considered a true positive:

At an IoU threshold of 0.5, predictions need to overlap with the ground truth by at least 50%.
At an IoU threshold of 0.95, the overlap requirement is much stricter, requiring a 95% overlap for a correct prediction.
By evaluating the model across multiple IoU thresholds, the mAP50-95 provides a comprehensive measure of the model's performance, capturing both precision (correctness of the predictions) and recall (coverage of all relevant instances).

Performance Results
The fine-tuned YOLOv8 model achieved a mean Average Precision (mAP50-95) of 0.7318 on the test set after 30 epochs. This score reflects the model's strong capability in accurately detecting and segmenting the target objects. The relatively high mAP score across a wide range of IoU thresholds indicates that the model is not only making accurate predictions but is also robust to variations in the overlap requirement, showcasing its generalization capability across different levels of object localization precision.


These are some detections predicted by the fine-tuned YOLOv8-segmentation visualized.
![Alt text](figures/prediction_test.jpg?raw=true "Sample prediction 1")
![Alt text](figures/prediction_test2.jpg?raw=true "Sample prediction 2")

# Reproducibility

In order to reproduce the results:
- Use train-test split of 80%-20%, random state 40.
- YOLOv8n-seg model from Ultralytics.
- 30 epochs of training.
- Use the libraries' version specified in 'Requirements' section.
- Use batch size of 4.
- Learning rate of 0.01 (default)


# To run the algorithm
Run: ./test_script.sh in order to run the algorithm, note that the algorithm assumes the position of the dataset being the location it is situated in Rangpur and the current location for training is '/home/Student/s4671748/comp3710-project/'. If the location differs, changes inside 'dataset.py' for data loading and 'train.py' must be made in order to run properly

## Acknowledgments

- ISIC 2018 Challenge organizers
- Ultralytics for YOLOv8
- Contributors to the ISIC archive
