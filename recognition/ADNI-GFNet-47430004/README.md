# Classification of Alzheimer's disease brain data using GFNet

## Author

Donghyug Jeong

Student ID: 47430004

## Project Overview

This project was developed as an attempt at a solution for Problem Number 5: "Classify Alzheimer’s disease (normal and AD) of the ADNI brain data using one of the latest vision transformers such as the GFNet set having a minimum accuracy of 0.8 on the test set."

The project uses the pytorch with GFNet model and the ADNI brain data set, and attempts to find the best combination of hyperparameters to maximise the accuracy of classification on the test set. In the process, it uses gfnet-xs architecture found in the original github repo [1].

## GFNet - Global Filter Network

Global Filter Networks is a transformer-style architecture, that uses a 2D discrete Fourier transform, an element-wise multiplication between frequency-domain features and learnable global filters, and a 2D inverse Fourier transform to replace the self-attention layer found in vision transformers [1]. According to Rao et al. [1], it "learns long-term spatial dependencies in the frequency with log-linear complexity".

The following is a gif created by Rao et al. [1] that demonstrates how GFNet works:
![intro](images/original_intro.gif)

## Global Filter Layer

GFNet consists of stacking several Global Filter Layers and Feedforward Networks [1].

## Why the PR has 2 LICENSE files

Since the original repo (found in [Inspiration](#inspiration)) used the MIT License, a copy of the MIT License has also been included in this sub-folder, while also containing the Apache license of Shakes' repo.

## Dependencies:

Older versions of below dependencies may work, but the following was the version used in the code, in conjunction with Python 3.12.4.

- pytorch: 2.4.1
- timm: 1.0.9 (requires searching on conda-forge)
- matplotlib: 3.9.2 (for plotting and visualising the data - actual model does not require it, but all .py files that can run import it)
- cuda: 12.6
- numpy: 1.26.3
- scikit-learn: 1.5.1
- torchvision: 0.19.1+cu118

## Structure of Data

The code requires that the directory structures to the images are as follows:

Note: "/home/groups/comp3710/ADNI/AD_NC/" represents the directory on the UQ HPC Rangpur. This directory may be changed in the get_dataloaders() function of [dataset.py](/recognition/ADNI-GFNet-47430004/dataset.py). After the AD_NC directory, the structure must be met (i.e. train/ and test/ must exist with AD/ and NC/ directories in each).

```
/home/groups/comp3710/ADNI/AD_NC/
                                 train/
                                        AD/
                                            image.jpeg
                                        NC/
                                            image.jpeg
                                 test/
                                        AD/
                                            image.jpeg
                                        NC/
                                            image.jpeg
```

An image in the data set may look like this (This is an image if the NC set, within the training set):

<p align="center">
    <img src="/recognition/ADNI-GFNet-47430004/images/Sample_train_data_808819_88.jpeg" alt="Example ADNI brain data">
</p>

## Inspiration

Significant portions of the code were taken from the following github repo:
https://github.com/shakes76/GFNet

This github repo is a fork of the official github repo of the original GFNet code by the authors of “GFNet: Global Filter Networks for Visual Recognition” [1].

## Hyperparameters

## Official References/Bibliography

[1] Y. Rao, W. Zhao, Z. Zhu, J. Zhou, and J. Lu, “GFNet: Global Filter Networks for Visual Recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10960–10973, Sep. 2023, doi: 10.1109/TPAMI.2023.3263824.
