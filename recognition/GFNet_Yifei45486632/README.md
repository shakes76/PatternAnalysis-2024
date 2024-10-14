# COMP3710 Project
## 1. Project Overview
This project is to classify brain MRI scans from the ADNI dataset into two categories: Cognitive Normal (CN) and Alzheimer’s Disease (AD). And the goal is to train a Vision Transformer model (GFNet) on these images and achieve a minimum classification accuracy of 0.8 on the test set.
### Algorithm Implemented
GFNet, a recent breakthrough in vision transformers, presents an advanced architecture for image classification by leveraging the power of Fourier transforms for efficient feature extraction. The GFNet model consists of two key components: feature extraction and classification. The feature extraction part transforms input images into feature maps using global Fourier transformations, allowing the network to capture both local and global patterns efficiently across various scales. Unlike traditional convolutional networks that rely on spatial convolutions, GFNet bypasses these operations by directly operating in the frequency domain, which reduces computational overhead while maintaining accuracy.
## 2. Data Description
### Data Sources
The dataset is provided by Alzheimer's Disease Neuroimaging Initiative (ADNI), with permission to use it from the EECS COMP3710 team at the University of Queensland[^1]. The data has been devided into two floders: train and test, and devided into CN and AD two kinds of data in each folder.
### Data Preprocessing

### Data Size

### Data Categorical Distribution

## 3. 模型结构
### 模型的设计和实验, 模型的结构 (使用的layers, 激活函数, 优化器)
## 4. 模型训练
### 模型训练的过程和结果，概述训练参数，遇到的挑战（具体的），解决方案
回调函数的设置
## 5. 评估结果
### 列出
## 6. Reference
[^1]: “Alzheimer’s Disease Neuroimaging Initiative,” ADNI. https://adni.loni.usc.edu/