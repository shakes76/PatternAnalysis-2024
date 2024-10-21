# Implementation of Computer Vision Neural Network for ADNI Dataset (Problem 5)

## Introduction

This repository contains code to train computer vision neural network designed to analyze images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to assist in the classification and understanding of Alzheimer's disease progression through deep learning techniques. This repository folder contains an implementation of [GFNet](https://ieeexplore.ieee.org/document/10091201).

## About the Model

[GFNet](https://ieeexplore.ieee.org/document/10091201) is a cutting-edge vision transformer neural network that prizes itself on efficiently capturing spatial interactions through its use of the fast fourier transform.

![GFNet Structure](https://github.com/user-attachments/assets/b8e67323-a4d2-4427-ac7c-0e3720ccc62a)

GFNet adapts the well-known vision transformer (ViT) models by replacing the self-attention layer with a global filter layer. 
GFNet contains:
- **Patch Embedding**: Initial input images are split into several smaller size patches, which are then flattened into a lower-dimensional space.
- **Global Filter Layer**: The fast fourier transform is used to find the spatial interactions between the data.
- **Feed Forward Network**: A Multi-Layer Perceptron processes the results from the global filter layer through a activation function to  learn non-linear transformations, improving the model's ability to learn features
- **Global Average Pooling and Classification**: After the N sections of Global Filter Layer & Feed Forward Network, the resulting information is pooled together then used to classify.
- **Normalisation layers**: Optional normalisation layers to normalise the values between each section, helping improve generalisability.


### Model Architecture
- **Base Model**: GFNet
- **Input Shape**: 240x240
- **Output Classes**: Alzheimer's (AD), Normal Cognition (NC)
- **Framework**: PyTorch

### How to use

# Training
Parameters
```zsh
python ./recognition/GFNet_s4641938/train.py [IMAGESIZE] [EPOCHS] [ROOTDATAPATH]
```

Example
```zsh
python ./recognition/GFNet_s4641938/train.py 240 50 ./ADNI/AD_NC
```

During training, the ongoing best model will be saved at ./recognition/GFNet_s4641938/best_model.pth

# Testing
After training
Parameters
```zsh
python ./recognition/GFNet_s4641938/predict.py [IMAGESIZE] [MODELPATH] [ROOTDATAPATH]
```

Example
```zsh
python ./recognition/GFNet_s4641938/train.py 240 ./best_model.pth ./ADNI/AD_NC
```

## Training Details

### Dataset
- **Source**: Alzheimer's Disease Neuroimaging Initiative (ADNI)
- **Training** 25120 (256x240) images
- **Test** 9000 (256x240) images
- **Preprocessing**: Images were resized, normalized, converted to grayscale, and augmented to enhance model robustness.
- **Train/Validation Split from Training data** 90/10
- **Train/Validation/Test Split** 66.26/7.36/26.37

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Epochs**: 50
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Loss Function**: Cross-Entropy Loss

### Training Procedure
1. **Load the Dataset**: Use `torchvision.datasets` and `torch.utils.data.DataLoader` to load and preprocess the ADNI dataset using torchvision transforms.
2. **Define the Model**: Instantiate the GFNet model following the given model parameters.
3. **Train the Model**: Execute training loops, monitor accuracy and loss, saving the best performing model after each epoch.

### Dependencies
- **Python**
- **PyTorch**
- **torchvision**
- **timm** >= 1.8.0

### Performance & Results
The model achieved the following results on the validation set:
- **Test Accuracy**: 0.635%
- **Learning Rate**: 0.0001
- **Epochs**: 50

## Loss/Accuracy Plot
![accuracyPlot](https://github.com/user-attachments/assets/ef0e3191-245a-4026-a393-0347dc81562c)

Here it is evident that the current model suffers from a significant amount of overfitting. 
In the report documentation below I discuss the methods I used to attempt to improve/fix this.
Inherently, this would be a large issue given the size of the dataset. ViT models are built to derive
spatial and image information from extremally large datasets (often 1+ million), while the training data
available only contains around ~25000 images. 

# Report & Process Documentation
###
Given the scale of the problem, while it is possible to add several sets of GFNet blocks together, the first choice was to try 1 GFNet block given the scale of the dataset and problem. Some of the important initial parameters used were:
- **Learning Rate**: 0.01
- **Epochs**: 100
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

A learning rate of 0.01 caused the loss to significantl oscillate during training, and struggled to converge, so the learning rate was reduced to 0.001, then 0.00001.

100 epochs resulted in a model that was capable of perfect accuracy on the training set, but only 54% on the test set, which resulting in a far too overfitted model - therefore the epochs were reduced and models were saved based on the best accuracy at the time.

When the overfitting was still a notable issue, Adam was changed to AdamW to attempt to help the model generalise by making it choose the parameters that generalised the model best.

### Challenges
The main challenge with this model has been overfitting.
Overfitting can easily be seen on the plot above:
(training accuracy goes to ~1.0, while test accuracy stagnates around ~0.62).

Various methods into data pre-processing and model scaling were investigated:
- **Pre-processing**: Modifying the training data by adding Gaussian noise, Random Erasure, Color Jitter
- **Model**: Mixture of complex multi-layer GFNet Pyramid models to simplistic single-layer GFNets
- **Dropout**: Adding dropout and normalisation layers and tested various levels

More complex models did no help convergence, since the issue was overfitting.

Throughout the process I added various different levels of dropout: In GFNet there are 2 values for GFNet (drop_rate and drop_path_rate).
'drop_rate' refers to a flat dropout rate in each global filter layer, while drop_path_rate refers to a linearly increasing drop rate. 
Updating the drop_rate or drop_path rate did not increase the overfitting issue.
There were additional transformations made to the images: random erasure, color jitter, gaussian noise.
None of the changes made any notable improvement to the test accuracy. 

Unfortunately, this leads into a major issue of a vision transformer mentioned before. 
It expects a significant amount of divergent data to train on. The original version of GFNet as originally implemented was built to handle
a subset of imageNet, which contains 14 million labelled images in total. The subset used for the model contains 1000 categories, 100000 training images and 50000 test images. The image information GFNet is expected to learn has significantly more variety in image appearence (dog, cat, house, tree,...) than for the ADNI dataset, containing 25000 MRI brain scan slices. 

Unfortunately none of the attempted changes were able to bring improvements to the overall test performance.

To test if overfitting could be stopped, I shrunk the size of the GFNet to tiny sizes:
- **Pre-processing**: Gaussian noise, Random Erasure, Color Jitter
- **Pre-processing**: Gaussian noise, Random Erasure, Color Jitter

This model produced the following results:


There is a small reduction in overfitting, but the overall accuracy does not improve. 





To help benefit any further research or investigation into this model, any further work on this model's performance would benefit with this in mind. 

## License
MIT License

### Acknowledgments
Many thanks to the Alzheimer's Disease Neuroimaging Initiative for creating the dataset used in this project.

An additional thanks to [Chandar Shakes](https://github.com/shakes76) to providing the cleaned ADNI brain dataset used in training this model. 

The file modules.py contains the GFNet implemented as desribed in the paper [Global Filter Networks for Image Classification](https://arxiv.org/abs/2107.00645) by Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou ([GitHub](https://github.com/raoyongming/GFNet))