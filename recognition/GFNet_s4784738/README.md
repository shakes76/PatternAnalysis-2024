# Classification of Brain Images in the ADNI dataset using the GFNet Vision Transformer
### Benjamin Thatcher/s4784738

## Classifying Brain Images
The ADNI dataset contains thousands of human brain images. Some of them are from people who have ALzheimer's disease and some are from people who don't. Early detection of ALzheimer's disease can greatly improve outcomes for patients, so it possible that machine learning algorithms may be able to improve our current early detection systems. The Global Filter Networks for image classification, or GFNet for short, is a powerful new image classifier. Using a GFNet, I classified the brain images in the ADNI dataset as Alhiemer's positive or negative.

## How a GFNet Works
The Global Filter Network is ...
~ Picture of GFNet architecture here ~

## Dependencies
To run this code, the following dependencies must be installed:
python
torch
numpy
matlab
timm
...

## Running the Code
The model can be trained by running 'python train.py'
By running 'python predict.py', you can ...

Configuration settings and hyperparameters can be found and modified in ...

## Plots
~ Accuracy and loss of training ~
~ Accuracy and loss of validation ~ 

## Reproducability
To make the training reproducible, it is possible to set a seed at the start of the training loop. This could be done as follows:

## Data Pre-processing
The images in the ADNI dataset are pre-processed in dataset.py. This process involves resizing them to 224x224 pixels, converting them to 3-chanel greyscale, and normalizing them. The training and validation splits recieve additional augmentations to help the model learn their features. These augmentations include random horizontal flips, vertical flips, and rotations.
Images in the train directory of the AD_NC folder where used to construct the training and validation splits, while test directory was the source of the testing split. This ensured that model could not learn the features of the testing data beforehand. The training-validation split I chose was 80-20.
