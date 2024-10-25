# Improved 3D Unet for Prostate MRI Segmentation

## Overview

The implementation of an improved 3D UNet segmenting the (downsampled) Prostate 3D data set with all labels having a minimum Dice similarity coefficient of 0.7 on the test set.The data is augmented using the appropriate transforms in PyTorch and is loaded from a Nifti file format ".nii.gz"


## Improved UNet3D Model

The Improved 3D UNet model is a CNN for 3D image segmentation.
The model is based on the keras brats UNet model using an encoder and decoder with a bottleneck inbetween. The encoder applies convolutions to the input images and reduces its dimensions to identify patterns in the images. The decoder applies up-convolutions to the downsampled images while concatenating at each step  with the relative encoder convolution in its respective layer. The bottleneck is the connection between the encoder and the decoder.

The dataset is split into training data, validation data and testing data. 
These datasets are a subset of the total data loaded, split into 70% training, 20% validation and 10% testing.

Training the model has the goal of minimising the DSC loss on the training data, used to maximise
the DSC in evaluation mode on the validation set. Appropriate transformations are applied to the 
data to maximise the speed of training.

## Dependencies

PyTorch version: 2.4.0
Numpy version: 1.26.4
Tqdm version: 4.66.4
Nibabel version: 5.3.1
Python version: 3.12.3

## Results