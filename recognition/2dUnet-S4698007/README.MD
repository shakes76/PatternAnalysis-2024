# 2D U-Net for Prostate MRI Image Segmentation

## Description

This project implements a 2D U-Net model for segmenting magnetic resonance imaging (MRI) scans from the **HipMRI study**. The main objective is to accurately detect and segment the prostate, which is a critical task in prostate cancer diagnosis. U-Net, a popular architecture for biomedical image segmentation, is used to classify six different tissue types in MRI slices, focusing on identifying the prostate for cancer detection.

The U-Net model learns to predict pixel-wise segmentation masks that label different tissue classes. It operates by progressively downsampling the input MRI slices to capture abstract features and then upsampling to restore spatial resolution, enabling accurate segmentation.

## How It Works

The architecture of U-Net is based on an **encoder-decoder** design. The encoder path uses convolution and pooling layers to capture high-level features from the input MRI image, reducing its resolution. The decoder then upscales these features using transposed convolutions and reconstructs the original spatial dimensions to produce a segmentation map. **Skip connections** between the encoder and decoder help preserve fine details by passing high-resolution features from the encoder to the decoder.

Once trained, the model takes MRI slices as input and outputs segmentation masks.

## Dependencies

To reproduce the results in this project, you will need the following dependencies:

- Python 3.11
- PyTorch 2.0.1
- Torchvision 0.15.2
- Nibabel 5.1.0
- Numpy 1.25.0
- Matplotlib 3.7.2
- Tqdm 4.66.1