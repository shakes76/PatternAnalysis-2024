# Detecting Skin Lesions using YOLO11
Since 2016, the International Skin Imaging Collaboration (ISIC) have run annual challenges with the aim of developing imaging tools to aid in the automated diagnosis of melanoma from dermoscopic images. This repository focuses on detecting skin lesions within dermoscopic images. This data could then be taken and used in furthor diagnostic tools and is a useful first step in achieving this. As specified by the task requirements, this repository makes use of YOLOv11 to perform image detection on the provided dataset by the ISIC.

## Dependencies
- PyTorch 2.5.1+cu124
- Torchvision 0.20.1+cu124
- Ultralytics 8.3.24
- Opencv-python 4.10.0.84

## About YOLO11
YOLOv11 is the 11th major iteration in the "You Only Look Once" family of object detection models and is a very recent development, having released in September 2024. YOLOv11 provides significant increases in accuracy and small object detection in comparison to previous versions, while still maintaining high efficiency and speed. The YOLO family of models are very flexible and can easily be trained to detect objects on a variety of custom datasets with little to occasionally no fine tuning needed provided that the dataset and labels are provided to it in the necessary formats.

### Model Architecture
YOLOv11 provides significant advancements over older versions of YOLO, with various improvements to the components that make up its architecture, but ultimately follows a very similar structure to its predecessors.

![Architecture Diagram of YOLOv11](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/YOLOv11Architecture.png)
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
|![Conv Features](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/stage0_Conv_features.png)|![SPPF Features](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/stage9_SPPF_features.png)|
|Attention Mechanism: Stage 10 - C2PSA Features| Final: Stage 22 - C3K2 Features|
|![C2PSA Features](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/stage10_C2PSA_features.png)|![C3K2 Features](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/stage22_C3k2_features.png)|

## About the ISIC2018 Dataset
### Dataset Breakdown
The [ISIC 2018 Task 1 Dataset](https://challenge.isic-archive.com/data/#2018) is comprised of 3694 dermoscopic full colour images broken down into three categories:
- 2594 Training Images
- 100 Validation Images
- 1000 Testing Images

Each of these categories is also accompanied by the same number of black and white Ground Truth masks.

|Dermoscopic Image|Ground Truth|
|---|---|
|![Dermoscopic Example](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/ISIC_0000000.jpg)|![Ground Truth Example](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/ISIC_0000000_segmentation.png)|

### Using the Dataset

To use the ISIC 2018 dataset with YOLOv11, preprocessing must be performed and the data must be arranged in the correct locations. To setup the dataset correctly follow the following steps.
1. Ensure that the variables *ISC2018_TRUTH_TRAIN*, *ISC2018_TRUTH_VALIDATE* and *ISC2018_TRUTH_TEST* point to the location where the appropriate Ground Truth files are stored.
2. Run dataset.py. This should create the required file structure for the YOLOv11 model and convert the ground truth images into labels formatted for use in YOLOv11.

Running dataset.py will produce the following file structure if it does not yet exist:
```
data
├── images
│   ├── test
│   ├── train
│   └── validate
└── labels
    ├── test
    ├── train
    └── validate
```
3. Copy the dermoscopic images into their respective test, train and validate folders.


## Training the Model
Once the dataset is correctly structured, the model is then able to be trained. The model was trained with hyperparameters fairly similar to YOLOv11's default values as a starting point, with plans to fine tune these values if needed:
- Epochs: 75
- Learning Rate: 0.01
- Momentum: 0.937
- Weight Decay: 0.0005
- Batch Size: -1 (This sets the batch size such that it consumes ~60% of total VRAM on the provided GPU, in this case 3.66GB on a RTX2060)

The model can be trained by running train.py.

### Training Results
The model took approximately 40 minutes to train and seemed to produce good looking results upon comparing a validation batch vs its labels.

|   |   |
|---|---|
|![Validation Batch Labels](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/val_batch1_labels.jpg)|![Validation Batch Predictions](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/val_batch1_pred.jpg)|

![Training Results](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/training_results.png)
The graphs refer to the following relevant metrics:
- box_loss: Box Loss - Error in the predicted boxes compared to labels
- cls_loss: Class Loss - Error in prediction of classes
- Precision: Proportion of True positive predictions vs All predicted positives, measures how prone the model is to false positives.
- Recall: Proportion of True positive predictions vs All true positives, measures the models ability to predict lesions.


Inspection of the final metrics on the validation sets after training completion showed the following.

|Box Loss|Class Loss|Precision|Recall|
|---|---|---|---|
|1.22463|0.45896|0.94926|0.96|

This was a significantly better result than expected given that the selected hyperparameters were intended to be a starting point that may need further tuning. As such it was decided just to leave these values as is due to the computational intensity of YOLO's tuning genetic algorithm and the strong performance displayed by the validation of training data not warranting the time required to manually tune these values.

### Tuning the Model
If you wish to do so, while for this run the model was not tuned further, you can get access to optimized hyperparameters by running tune.py. This file will run YOLOv11s hyperparameter optimizer that makes use of a genetic algorithm to determine the best hyperparameters for a given dataset. Be aware that even on powerful hardware such as the Nvidia A100, this process can take upwards of 6 hours even for small datasets.

### Evaluating Against the Test Set
It was also decided to validate the model against the set of test data, given that the dataset provided by ISIC contained ground truthes for these images as well. This was performed with an IOU threshold of 0.8 and confidence threshold of 0.5 as the predictions would also require these thresholds as specified by the task. This validation resulted in a precision of 0.965 and Recall of 0.939 for the model.
![Validations results of the modal against Test set](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/testvalidate.png)

If you wish to run this for yourself, simply run evaluate.py.

## Predicting Lesions
Once validation was complete and it was ensured that the model was performing well, it was then used to detect lesions in the test dataset.

Here is some predictions and their true labels to compare to:

|Prediction|Reality|
|---|---|
|![Model's Prediction](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/test_batch2_labels.jpg)|![Reality](https://github.com/LonelyNo/PatternAnalysis-2024/blob/topic-recognition/recognition/yolo-47450253/images/test_batch2_pred.jpg)|

To perform the prediction on the test set, simply run predict.py

## Reproduction

To reproduce the results seen above:
1. Follow the instructions in the *Using the Dataset* section above.
2. Run train.py and wait for training to complete.
3. Run predict.py to generate predictions on the test set.
4. (Optional) Run evaluate.py to generate the exact array of images seen in *Predicting Lesions* and view other metrics regarding the prediction process.

## Conclusion and Improvements
Overall, while model was able to successfully detect skin lesions within the ISC2018 test set to the required parameters of the task. While the selected hyperparameters were able to train the model such that it solved the task sufficiently, taking the time to run the tuning script to generate more optimal parameters would likely result in a model that is even better at the detection task, or at the bare minimum the ability to train the model in a lower number of epochs, thus speeding up training time.

## A note on the lack of modules.py
As mentioned in Ed Post #336, using ultralytics to perform this task is an allowed method and as mentioned in Ed Post #444, this means that all of the functionality that would be stored in modules.py is already provided by the pretrained model itself. As anything that would be added to this would essentially just be a simple wrapper for YOLO's methods I elected to not include this file.

## References
- [ISIC Challenge Website](https://challenge.isic-archive.com)
- [ISIC 2018 Task 1 Dataset](https://challenge.isic-archive.com/data/#2018)
- [YOLOv11 Architecture Explained: Next-Level Object Detection with Enhanced Speed and Accuracy - By S Nikhileswara Rao](https://medium.com/@nikhil-rao-20/yolov11-explained-next-level-object-detection-with-enhanced-speed-and-accuracy-2dbe2d376f71)
- [YOLOv11: An Overview of the Key Architectural Enhancements - By Rahima Khanam and Muhammad Hussain](https://arxiv.org/html/2410.17725v1)
- [YOLOv8 Architecture Explained! - By Abin Timilsina](https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5)
- [Ultralytics Docs: Performance Metrics Deep Dive](https://docs.ultralytics.com/guides/yolo-performance-metrics/#object-detection-metrics)