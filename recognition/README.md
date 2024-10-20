# Hip MRI Segmentation Using 2D U-Net 
## Description of Algorithm
This project implements a 2D U-Net model for segmenting MRI scans of the hip. The problem being solved is the segmentation of regions in MRI images, specifically identifying and separating target areas (such as muscles or bones) from the surrounding tissues.

In the example below, the left side shows an input MRI image, and the right side shows the corresponding segmentation mask produced by the U-Net model. The goal is to label each pixel as either part of the region of interest (white) or as background (black).

![Input MRI Image](images/image_mask_0.png)

## How the 2D Unet works

The U-Net model is designed for image segmentation tasks and follows an encoder-decoder architecture with skip connections between corresponding layers in the encoder and decoder. The key steps of the U-Net model are:

1. **Contracting path (Encoder)**: The input MRI image is passed through multiple convolutional and max-pooling layers to capture spatial context and reduce the dimensionality of the feature maps. In the case of my Unet, there will be 4 of these convolutional blocks, each with 2 convolutional layers using ReLU activation and the second with batch normalisation, and a max pooling layer.
2. **Bottleneck**: At the bottleneck, the deepest part of the network, the model captures the most abstract and meaningful features.
3. **Expanding path (Decoder)**: The feature maps are then upsampled using transposed convolutions, and merged with high-resolution features from the encoder to recover spatial details.
4. **Output**: The final layer uses a sigmoid activation function to generate a binary segmentation mask where each pixel is classified as part of the target region or background.

These key segments of the model can be seen in the diagram below (REF).
![Unet-Architecture](images/unet_architecture.png)




## Dependencies 