# GFNet-ADNI
This repository contains the code for training a custom-made GFNet model used to identify Alzheimer's disease in 2D sliced MRI brain scans. The model was trained on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, which contains a number of sliced MRI brain scan images categorized into Cognitive Normal (NC) and Alzheimer's Disease (AD) classes. The model architecture is based on GFNet, a convolutional and global filtering-based architecture designed to efficiently extract spatial patterns from images for classification.

## The GFNet Algorithm
GFNet is a powerful neural network architecture that combines convolutional layers with global filters, allowing it to capture both local and global image features effectively. Unlike traditional CNNs that focus only on local features, GFNet can process the entire image context using global Fourier-like filtering mechanisms, which enhances its ability to recognize complex patterns across the whole image.

