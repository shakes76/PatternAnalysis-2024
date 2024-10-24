
# 3D Improved UNet for Prostate Segmentation

## Task description:
This project implements a 3D Improved UNet model to segment prostate regions from downsampled MRI scans. 
The goal is to achieve a minimum Dice similarity coefficient of 0.7 on the test set for all labels.
One of the datasets used was from the HipMRI study and contained prostate 3D MRI data from 38 patients (for the 3D task), totalling 211 3D MRI volumes. I need to segment six categories, which are background, body contour, bone, bladder, rectum and prostate region.
## Model Description
3D UNet is based on the standard 2D UNet architecture, but extended to handle 3D volumetric data. The network consists of an encoder part and a decoder part, the encoder part is used to analyse the whole input image, it contains four resolution steps, in each of which two 3×3×3 convolution operations are used, followed by a corrected linear unit (ReLU), and a 2×2×2 maximal pooling with a step size of 2 is used to reduce the feature map size.
The decoder part, on the other hand, is responsible for generating the full-resolution segmentation output for reconstructing the segmented full-resolution image. Each layer consists of a 2×2×2 upsampling convolution with a step size of 2, followed by two 3×3×3 convolutions, each followed by a ReLU as well.Each layer in the decoder has jump connections to the corresponding resolution layer in the encoder path to provide the necessary high-resolution features to aid in the reconstruction. The main improvement is the replacement of all 2D operations with their 3D counterparts, such as 3D convolution, 3D max-pooling and 3D upsampling of the convolutional layer weight loss function and a special data enhancement strategy that makes it possible to train the network with only a few manually labelled slices in order to generate a full 3D volume segmentation . As can also be seen in the figure below, the encoder and decoder sections need to be symmetric, a structure that allows high-resolution features to be passed directly to the decoder to help preserve more detail as the resolution is gradually recovered. Each convolutional module is immediately followed by a jump connection that connects the encoder output to the decoder input at the same resolution. This is important to preserve spatial detail in the segmentation results.
<div align="center">
  <img src="unet architecture.png" >
</div>
**attention** we cite the picture from 3DUnet[1]

## Project Structure
- `modules.py`: Contains the implementation of the 3D Improved UNet model.
- `dataset.py`: Loads and preprocesses the downsampled prostate MRI dataset.
- `train.py`: Scripts for training, validation, testing, and saving the model.
- `predict.py`: Provides inference examples and visualization of segmented regions.
- `README.md`: This file, documenting the project.

## Dependencies
- Python==3.11
- torch==2.0.1
- monai==1.3.2
- numpy==1.26.4
- matplotlib==3.9.2
- nibabel==5.3.0
- scikit-learn==1.5.1

Use the following command to create the conda environment
```
conda env create -f environment.yml
```
activate environment
```
conda activate pytorch-2.0.1
```