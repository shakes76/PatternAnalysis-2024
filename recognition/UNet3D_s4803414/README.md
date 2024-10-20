# 3D U-Net for MRI Segmentation
## Description
This repository contains a PyTorch implementation of a 3D U-Net
model for semantic segmentation of MRI images. The algorithm is 
designed to accurately segment medical images by predicting 
pixel-level masks. The problem solved by this algorithm is 
vital in medical image analysis, where precise segmentation 
helps in diagnosing conditions, planning treatment, and 
evaluating therapy outcomes. Specifically, this implementation 
focuses on semantic segmentation of MRI scans from the HipMRI 
Study dataset. 


## How It Works
The 3D U-Net architecture is a convolutional neural network 
designed to operate on 3D input volumes, making it particularly 
effective for volumetric medical data such as MRI scans. The 
model follows an encoder-decoder structure, where the 
**encoder** compresses the input into a latent representation, and 
the **decoder** reconstructs it into a segmentation mask. **Skip 
connections** between corresponding layers in the encoder and 
decoder help retain important spatial information lost during 
compression. This network outputs a class label for each voxel 
(3D pixel), predicting structures like organs, lesions, or other 
relevant features.

The diagram below illustrates the 3D U-Net architecture, showing 
how the model processes input volumes and generates segmentation 
masks:

![3D U-Net Architecture](./content/UNET_model.png)

### Visualization
Below is an example visualization of the segmentation process for a given slice from an MRI scan:
- **Left**: Original MRI slice
- **Center**: Ground truth segmentation mask
- **Right**: Model prediction

![Sample Visualization](./visuals/visualization_sample.png)


## Dependencies
To run the code, you need the following dependencies:
* Python 3.12
* PyTorch 2.0.0
* Torchvision 0.15.0
* Matplotlib 3.8.0
* Nibabel 5.2.0
Make sure to install these using `pip` or `conda`. For example:

```bash
pip install torch torchvision matplotlib nibabel
```

## Reproducibility
To ensure reproducibility, all random seeds are set explicitly 
within the training script. If you want to reproduce the 
results exactly, ensure the same dataset splits, 
hyperparameters, and model initialization are used. The model 
weights are stored and can be reloaded from `model/new_model.pth`.

## Example Inputs and Outputs

### Input
The input to the model is a batch of MRI images in NIfTI 
format, typically with dimensions `(depth, height, width)`. 
These are loaded using the `nibabel` library and processed as 
3D arrays.

**Example Input:**
- MRI Volume: `Case_040_Week5_SEMANTIC_LFOV.nii.gz`

### Output
The model outputs a 3D segmentation mask, where each voxel is 
assigned a class label corresponding to the target structure.

**Example Output:**
- Segmentation mask: A 3D array of dimensions `(depth, height, width)` with integer values representing class labels.

Below is a sample input MRI slice and the corresponding 
segmentation output.

- **Original Slice:**
![Original Image](./visuals/visualization_original.png)
- **Prediction:**
![Prediction Mask](./visuals/visualization_pred.png)

## Pre-processing
Before training, the MRI volumes were normalized to ensure 
consistent intensity distribution across the dataset. 
Additionally, each MRI scan was resized to a fixed resolution 
to match the input dimensions required by the 3D U-Net. Data 
augmentation techniques such as random rotations, flipping, 
and cropping were applied to increase the diversity of the 
training data.

## Data Splitting
The dataset was split into **training (80%)**, 
**validation (10%)**, and **testing (10%)** subsets. 
This split was chosen to ensure a sufficient amount of data 
for training while reserving enough data for accurate 
evaluation. The data was stratified by cases to ensure that 
images from the same subject were not present in multiple 
sets, which could otherwise lead to overfitting.

## References
- Isensee, F., et al. "nnU-Net: a self-adapting framework for U-Net-based medical image segmentation." Nature Methods 18.2 (2021): 203-211.