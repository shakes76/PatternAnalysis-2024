# Prostate 3D data Segmentation with 3D-UNet and Improved 3D-UNet models
## Student information
**Name:** Sara Alaei

**Student number:** 47041437

**Task:** Taks #4 

## 3D U-Net and Improved 3D U-Net architectures 
The use of both the original 3D U-Net by Ozgün Çiçek et al. and the improved 3D U-Net model by Fabian Isensee et al. were investigated within this report. One of the main applications of the 3D U-Net model is medical image segmentation, specifically targeting volumetric data such as MRI scans. It addresses the challenge of identifying anatomical structures or lesions in these images, aiding in the diagnosis, treatment, and monitoring of diseases. 

The original U-Net architecture is a powerful convolutional neural network, featuring a symmetrical structure that includes encoder and decoder paths. The encoder progressively down-samples the input through various convolutional layers to capture context and enable the model to learn the varied features present. The output from the encoder is then reconstructed in the decoder path, wherein skip connections from the encoder enables the model to retain the spatial information and produce accurate segmentations of bodies within the images. 

The improved U-Net is a more complex and modern model that enhances the foundational design of the original U-Net. Improved 3D U-Net uses a deeper architecture with multiple convolutional layers, which enables the model to learn intricate pattern and relationships in the data more effectively. Furthermore, the inclusion of batch normalisation, improved skip connections and up-sampling aid in the stabilisation, generalisation, and the transfer of finer details. Therefore, these enhancements in the improved U-Net model leads to more sophisticated, accurate and high-quality segmentations compared to the original U-Net.

INCLUDE IMAGE

## Algorithm (pre-processing, training and inference)

## Parameters and Results

## Dependencies


## References