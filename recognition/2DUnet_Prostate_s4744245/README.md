# 2D Unet on HIPMRI Study on Prostate
## Description
This repository is an implementation of a 2D Unet applying segmentation on prostate scans from CSIRO study. 
The Unet segments the MRI images into following classes:
* Class 0: Background
* Class 1: Body
* Class 2: Bones
* Class 3: Bladder 
* Class 4: Rectum
* Class 5: Prostate

  
The quality of the segmentation is measured using DICE Coefficents for each class with the goal of achieving a minimum of 0.75 for each class.
The project shows a real world applcation of a Unet n a medical setting, being capable of idetifying different sections of MRI images taht could help
doctors save time and effort in the analysis of medical results.

## The 2D Unet
A 2D Unet is a type of convolutional neural network, primaril used for image segmentation with a sturture similar to below:
### 2D Unet Architecture
![image](https://github.com/user-attachments/assets/acd5c79d-1966-4033-abdd-e856c9f796c1)

### Characteristics of a 2D UNET
As seen in the image above, the primary characteristic of teh Unet s the encoded and decoder in a "U" shape with skip connections. The encoders, on the left, are the contracting path down to the bottleneck
where the spatial dimension is the smallest and the decoder, on the right, reconstructs the image for the segmentation output. The skip connects are what make the Unet unique. The skip connections pass
the features from the endocer layers to the corresponding decoder layer, retaining information that helps with specific detail in the segmentation that would be lost otherwise. 2D convolutions are used to pass the
feature imformation between layers and rasposed convolution to ensure the correct reshaping of the image. I have used RELU activation functions, He uniform initialisers and variable dropout through the model. 
The final layer of the model is a convultion using softmax activation for the final segmented output of the 6 classes.

## Loss Function
A weighted categorical cross-entropy loss function was used for training while the dice coefficient was used for validation.
The loss function is described below:

![image](https://github.com/user-attachments/assets/5cb0f5b1-27bf-44d9-b271-4bedacc7eb01)

For each class such that the label is correct.
The loss functin for my traning is displayed below:

![image](https://github.com/user-attachments/assets/33e74f04-3139-4932-a159-924c86f196a9)

--------------------------------------------------------------------------------------------------------

## Output
### Ouput 1
<img width="422" alt="image" src="https://github.com/user-attachments/assets/466cec37-428d-4994-8cbb-dd5196e84521">


### Ouput 2
<img width="422" alt="image" src="https://github.com/user-attachments/assets/a4c1b782-1213-470b-9240-0965a7018036">


### Ouput 3
<img width="428" alt="image" src="https://github.com/user-attachments/assets/6bb487e4-0ef1-41d6-90ad-e44d4b264c07">


Some expample of the output are displayed above. The Original images displays the MRI input image, the Ground Truth Mask is the true labels of segmentation that is the goal of the model
and the Predicted Mask is the model output. all three exmaples show goo dperformance with some mior imrpovement possible with specfific details, seen specifically with the smooth boundary edges
between classes for the predicted masks.

The Dice Coefficients for each class of my model are as shown below:

Dice Coefficient for Class 0: 0.9959
Dice Coefficient for Class 1: 0.9773
Dice Coefficient for Class 2: 0.9193
Dice Coefficient for Class 3: 0.9309
Dice Coefficient for Class 4: 0.7823
Dice Coefficient for Class 5: 0.7989

With all classes achieveing the goal of the 0.75 dice coefficient, the power of the Unet is displayed although there is certainly some imporvement possible, specifically with the
rectum and prostate labels.

## Prepocessing 



## Dependencies
* Python 3.9.18
* Tensorflow 2.11.0
* NVIDIA CUDA 11.8
* Numpy 1.23.5
* Matplotlib 3.9.2
* Glob
* Nibabel
* Tqdm

  
