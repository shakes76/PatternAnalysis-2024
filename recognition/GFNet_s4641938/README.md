# Implementation of Computer Vision Neural Network for ADNI Dataset (Problem 5)

## Introduction

This repository contains code to train computer vision neural network designed to analyze images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to assist in the classification and understanding of Alzheimer's disease progression through deep learning techniques. This repository folder contains an implementation of [GFNet](https://ieeexplore.ieee.org/document/10091201).

## About the Model

[GFNet](https://ieeexplore.ieee.org/document/10091201) is a cutting-edge vision transformer neural network that prizes itself on efficiently capturing spatial interactions through its use of the fast fourier transform. GFNet adapts the well-known ViT Transformer models by replacing the self-attention layer with a global filter layer. 

'SHOW GLOBAL FILTER'

### Model Architecture
- **Base Model**: GFNet
- **Input Shape**: 240x240
- **Output Classes**: AD, ND
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
- **Training** 25120 images
- **Test** 9000 images
- **Preprocessing**: Images were resized, normalized, and augmented to enhance model robustness.
- **Train/Validation Split from Train** 90/10
- **Train/Validation/Test Split** 66.26/7.36/26.37

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Epochs**: 50
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Loss Function**: Cross-Entropy Loss

### Training Procedure
1. **Load the Dataset**: Use `torchvision.datasets` to load and preprocess the ADNI dataset.
2. **Define the Model**: Instantiate the GFNet model.
3. **Train the Model**: Execute training loops, monitor loss, and save best performing model.

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


# Report & Process Documentation
###
I began by using the model code GFNet, attempting to run it using the Adam.

### Challenges
The main challenge with this model has been overfitting.
This can be seen on the plot of training versus test accuracy.

Various methods into data pre-processing and model scaling were investigated:
- **Pre-processing**: Gaussian noise, Random Erasure, Color Jitter
- **Model**: Mixture of complex GFNet Pyramid scaling models to simplistic GFNets
- **Dropout**: Adding dropout layers and adapting

More complex models did little to help with 

Throughout the process I added various different levels of dropout: In GFNet there are 2 values for GFNet (drop_rate and drop_path_rate).
'drop_rate' refers to a flat dropout rate in each global filter layer, while drop_path_rate refers to a linearly increasing drop rate. 
Updating the drop_rate or drop_path rate did not increase the overfitting issue.
There were additional transformations made to the images: random erasure, color jitter, gaussian noise.
None of the changes made any notable improvement to the test accuracy. 

Unfortunately, this leads into a major issue of a vision transformer. It requires a significant amount of data.
The training dataset used on this data is ~25000 images, which is much smaller than expected for a visual transformer
Vision transformers are built to specificially able to find spatial information in complex large datasets. 
For a small dataset like this ADNI subset it rapidly results in overfitting.

Unfortunately none of the attempted changes made any noticable improvements to the model. 
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
