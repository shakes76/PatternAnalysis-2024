# Use of YOLOv11 for Lesion detection in the ISIC2017 dataset 
## 1. Introduction to YOLOv11
The introduction of the You-Only-Look-Once (YOLO) algorithm for image detection was first released in 2016 by Joseph Redmon et al. - since then, numerous other researchers have worked on improving the networks' efficiency and accuracy which eventually led to the release of YOLOv11 by Ultralytics in 2024.

The YOLO architecture is a convolutional neural network that prioritises speed over accuracy by using a single-stage detection process, as opposed to the two-step process utilised by many of the more accurate object detection algorithms.

![Comparison of one and two stage detectors.](recognition/ISIC_47044090/figures/detectors.jpg)

YOLO's algorithm can be better understood by analysing its architecture:

![Diagram of YOLOv11 architecture.](/figures/architecture.webp)

Essentially, the convolution neural network parses over each pixel in the image and, in one step, tries to fit a bounding box with the pixel as the box's centre. In doing so, the model also outputs a confidence score for the validity of the proposed box. By the end of the single iteration through the image, the model outputs each proposed bounding box with a confidence score above a specific interval (0.25 by default for YOLOv11) as the final detections for the image.

This is in contrast to other two-stage models that split this process into two stages, which generally leads to improved accuracy at the cost of inference and training speed. This makes YOLO (and other one-stage models) ideal for fast-paced environments such as real-time detection or when resources are limited, while the two-stage models are generally preferred when accuracy is priority.

The newest innovation in the YOLO model is Ultralytic's YOLOv11, which is said to perform better and faster than the previous models. Ultralytics is also responsible for the release of every iteration since YOLOv8 in 2023.

## 2. Introduction to lesion segmentation and the ISIC2017 dataset
### 2.1 Lesion segementation
This algorithm aims to provide a fast and accurate form of lesion detection from a single dermoscopic image - in this case, the accuracy of detection and segmentation is paramount, as it would theoretically be used for the medical purposes.

![Example of lesion segmentation.](/figures/lesion_detection.png)

### 2.2 2017 ISIC Challenge
To train the model, a dataset from the 2017 ISIC challenge was used: this includes 
- 2000 training samples
- 200 testing samples
- 150 validation samples
This training validation split was used since it was the one provided by the ISIC challenge, and the approximately 80/10/10 split is seen to be optimal for large scale training. 

Each of the samples contains and image and a ground truth, where the image is a dermoscopic picture in .jpg file of varying sizes, and the ground truth is the corresponding mask in black and white which segments the image. These can be downloaded from the ISIC Challenge datasets website (2017), which is linked in Appendix 1.

![Example of ISIC image and masks.](/figures/ISIC_image_mask.png)

However, these cannot directly be plugged into YOLOv11, as Ultralytics' YOLO models require a specific file format and file structure. Firstly, it requires that all images are of equal and even dimensions. Secondly, all labels must be in the form of a .txt file which contains (for each bounding box identified): its class, centre x-coordinate, centre y-coordinate, width and height.


## 3. Processing dataset for YOLOv11 compatability
### 3.1 Process
As was discussed, the images in the ISIC 2017 dataset are of varying sizes and each have un-even dimensions. To alleviate this problem, each image was extended using a black letter-box on its smaller dimension (between x and y) until the image was a square. This method, while achieving the purpose of even dimensions in the image for YOLOv11 compatibility, ensures no warping occurs (which could hinder the training of the model).

![Example of resizing ISIC image.](/figures/original_modified.png)

After this point, the image was scaled down to 512x512, which was a picked as it serves as a midpoint between maximising information (larger image=more information=better training) and minimising computation time (larger image=longer training=more resources). This was applied to both the images and the ground truth masks provided by ISIC.

For Ultralytics compatability, the masks also needed their information transferred to a .txt file in the above-mentioned format. This was done by extracting the bounding box around coordinates who's pixel values were not 0 (as lesions are highlighted white and everything else is black in the masks provided). For this task, there is only one class to detect (lesion).

### 3.2 Organisation
This is all implemented in the dataset.py file, but to use this (and Ultralytics' YOLO models), the files must be organised in a specific way:

ORGANISATION OF FILES (ORIGINAL_ISIC/ISIC/etc)

### 3.3 dataset.py usage
Simply run the dataset.py to process all images and masks into the required format. Prior to training, create a .yaml file in the datasets folder that looks like:

YAML FILE

modifying any file directories if necessary.

## 4. Training model for lession detection
### 4.1 Running train.py
The train.py file contains three methods: run_train() and run_test(). By default, running the file runs both of them which trains the model (using YOLOv11 as a base) using the supplied data.

The training will output the results of training (graphs and .csv containing metrics) in a directory named runs/detect/trainX (where X is the train number). The trained model is located within this directory in the weights folder (under best.pt) - this can then be used to run further inference and testing on the model

IMAGE WITH EXAMPLE DIRECTORY

### 4.2 Results from training
The result graphs for training can be seen below:

GRAPHS

## 5. Testing model results
### 5.1 Running test.py
The file (by default) also then runs a test on the most recent training batches' weights (the one that was just trained if they are run together).

By running inference on each image in the dataset, it calculates intersection-over-union score indepenently (metric used to evaluate the allignment of predicted and true bounding boxes) before outputting the average and number of samples with IoU above 0.8.

### 5.2 Results from testing
In this case, resulting metrics were found to be:

IMAGE WITH AVE IOU AND N IOU>0.8

## 6. Predicting using trained model
predict.py contains a method to visualise inference (example usage) of the trained model. By default, it also runs on the most recent training batch's weights. It plots and displays 9 random samples of the image, the model's predicted lesion and the true lesion from the test set.

![Example of predict.py output.](/figures/predict_examples.png)

## 8. Appendix
### Appendix 1: ISIC Dataset
### Appendix 2: Requirements
### Appendix 3: References
image 1 https://www.v7labs.com/blog/yolo-object-detection 
yolo architecture and evolution https://medium.com/@nikhil-rao-20/yolov11-explained-next-level-object-detection-with-enhanced-speed-and-accuracy-2dbe2d376f71