# **Segmentation Using 2D UNet with Minimum Dice Coefficient of 0.75**
## Author
### Xinyue Yuan (Student ID: 48792286)
## Project Overview
#### The objective of this project is to segment prostate cancer lesions (https://doi.org/10.25919/45t8-p065) using a 2D UNet model. Accurate segmentation of these lesions is crucial for diagnosing and treating prostate cancer effectively. The goal is to achieve a minimum Dice similarity coefficient (DSC) of 0.75 on the test set for the prostate label.The project utilizes processed 2D slices in Nifti file format to train and evaluate the segmentation model.
## Environment Dependencies
### This packages/software need to be installed
#### python 3.12
#### numpy 1.26.4
#### matplotlib 3.8.4
#### torch 2.3.1
#### Nibabel 5.3.1
## Repository Layout
### dataset.py contains the data loader for loading and preprocessing data
### modules.py contains the source code of the components of model
### predict.py contains example usage of  trained model 
### train.py contains the source code for training
## DataSet
### Here is an example
### !(Example)[images/img.png]
#### Each label file is in NIfTI format (.nii or .nii.gz) and contains annotated information for different slices regarding the prostate and surrounding tissues
### Data Processing
#### 3D medical images are converted into multiple 2D slices so that the model can be processed one by one；the image data is normalized to ensure that the value range of the input data is consistent
## Model
#### UNET is a U-shaped encoder-decoder network architecture, which consists of four encoder blocks and four decoder blocks that are connected via a bridge. The encoder network (contracting path) half the spatial dimensions and double the number of filters (feature channels) at each encoder block. Likewise, the decoder network doubles the spatial dimensions and half the number of feature channels.[1]
### ![UNET ARCHITECTURE](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*lvXoKMHoPJMKpKK7keZMEA.png)
###
## Results
###
## Conclusion
###
## Discussion
### 
## References
### [1]https://medium.com/analytics-vidhya/what-is-unet-157314c87634

