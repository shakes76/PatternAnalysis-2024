# Implementation of Computer Vision Neural Network for ADNI Dataset (Problem 5)

## Introduction

This repository contains code to train computer vision neural network designed to analyze images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to assist in the classification and understanding of Alzheimer's disease progression through deep learning techniques. This repository folder contains an implementation of [GFNet](https://ieeexplore.ieee.org/document/10091201).

## About the Model

[GFNet](https://ieeexplore.ieee.org/document/10091201) is a cutting-edge vision transformer neural network that prizes itself on efficiently capturing spatial interactions through its use of the fast fourier transform. GFNet adapts the well-known ViT Transformer models by replacing the self-attention layer with a global filter layer. 

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
- **Preprocessing**: Images were resized, normalized, and augmented to enhance model robustness.

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

### Challenges
The main challenge with this model has been overfitting.
This can be seen on the plot of training versus test accuracy.

Various methods into data pre-processing and model scaling were investigated:
- **Pre-processing**: Gaussian noise, Random Erasure, Color Jitter
- **Model**: Mixture of complex GFNet Pyramid scaling models to simplistic GFNets

To help benefit any further research or investigation into this model, any further work on this model's performance would benefit with this in mind. 

## License
MIT License

### Acknowledgments
Many thanks to the Alzheimer's Disease Neuroimaging Initiative for creating the dataset used in this project.

An additional thanks to [Chandar Shakes](https://github.com/shakes76) to providing the cleaned ADNI brain dataset used in training this model. 

The file modules.py contains the GFNet implemented as desribed in the paper [Global Filter Networks for Image Classification](https://arxiv.org/abs/2107.00645) by Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou ([GitHub](https://github.com/raoyongming/GFNet))


