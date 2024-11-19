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

Each image was preprocessed into bounding boxes from the given ground truth masks, normalized for size consistency, and stored in YOLO-readable formats.

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

## Training
To Train the model we must have all the neccesary data.
Data augmentation (flipping, scaling, and cropping) was applied to improve generalization. The data was already split as:
- **70%** for training (images and masks)
- **10%** for validation (images and masks)
- **20%** for testing (images and masks)

<div align="center">

| **Train Batch 1** | **Train Batch 2** | **Train Batch 3** |
|-------------------|-------------------|-------------------|
| ![Train Batch 1](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/train_batch0.jpg) | ![Train Batch 2](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/train_batch1.jpg) | ![Train Batch 3](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/train_batch2.jpg) |

</div>

The model learns on the images and given masks seen above. However, the dataset is not very simple so we need to find suitable parameteres.

#### Initial Training Results
The first round of training revealed opportunities for optimization. Below are the initial loss metrics recorded:
- **Box Loss**: 1.15822 Measures the error in predicted bounding boxes.
- **Class Loss**: 1.7165 Ensures correct class prediction.
- **Distribution Focal Loss (DFL)**: 1.40207 Helps with localization accuracy.

#### Hyperparameters Used for Fine-Tuning
To further improve the model’s performance, the following hyperparameters were adjusted:
- **Optimizer**: AdamW  
- **Batch Size**: 16  
- **Learning Rate**: 0.001  
- **Weight Decay**: 0.0005  
- **Epochs**: 75 

#### Fine-Tuned Loss Metrics
After fine-tuning, the model demonstrated significant improvement, with the following loss values recorded:
- **Box Loss**: 0.54229
- **Class Loss**: 0.30157
- **Distribution Focal Loss (DFL)**: 0.97581

## Validation
Validation metrics were monitored throughout training on the 150 images to ensure the model wasn't overfitting.

- **Precision**: 0.872   Precision measures how many predicted positive were actually correct.
- **Recall**: 0.860   Recall measures the proportion of actual positives that were correctly identified by the model.

![Plotted Results from Training](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/results_train.png)

The training and validation curves allow us to analyse model performance over time. A good model should be decreasing loss values in both the training and validation sets. Showing that the model is learning effectively without overfitting. The box loss measures the difference between the predicted and actual bounding box coordinates the lower the best as it corralted to a higher Iou when decreasing . 


## Testing
The model was tested on the 1200-image test set, with an IoU threshold of 0.8 ensuring high quality predictions.

### Results  

#### Average IoU Recieved:
    IoU avg: 0.815

#### Metrics Recived:
| Metric      | Value  |
|-------------|--------|
| Precision   | 0.855  |
| Recall      | 0.873  |



The model has an average of 0.815 Iou on the test set confirming that the model produces bounding boxes with high overlap against the ground truth. The 87.3 % suggests that the model is classing the images well.


![Test Predictions](https://github.com/mraula/PatternAnalysis-2024/blob/topic-recognition/recognition/YOLOs4703527/figures/test_pred.jpg)


The figure above show the sample test predictions where the bounding boxes generated by the model are visualised. These results show that the model segments the images well with the predicted labels aligning closely with the ground truth labels. The predictions reflect the model's ability to capture lesion regions accurately even on challenging samples from the test set.

## Usage and Dependencies
### Folder Structure
```bash
├── train.py            
├── predict.py          
├── modules.py        
├── data                
│   ├── train          
│   │   ├── images     
│   │   └── labels     
│   ├── val          
│   │   ├── images     
│   │   └── labels    
│   └── test          
│       ├── images    
│       └── labels  
└── results           
```


### Dependencies
- Python 3.12.7  
- torch 2.5.0+cu118  
- torchvision 0.20.0  
- ultralytics 8.3.21  
- OpenCV 4.10.0  

#### Install dependencies:
```bash
pip install torch torchvision ultralytics opencv-python
```

## Example Usage
### Preprocess Data
```bash
python dataset.py
```
Preprocess the dataset if you require and insure you have the correct folder structure.
### Train the Model
```bash
python train.py
```
### Run Predictions
```bash
python predict.py
```

## References

- [Detailed Explanation of YOLOv8 Architecture (Part 1)](https://medium.com/@juanpedro.bc22/detailed-explanation-of-yolov8-architecture-part-1-6da9296b954e)  
- [What is YOLOv8?](https://blog.roboflow.com/what-is-yolov8/)  
- [YOLOv8 Architecture](https://yolov8.org/yolov8-architecture/)  
- [Fine-tuning YOLOv8 Using a Custom Dataset](https://medium.com/@yongsun.yoon/fine-tuning-yolov8-using-custom-dataset-generated-by-open-world-object-detector-5724e267645d)  

