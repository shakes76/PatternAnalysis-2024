# COMP3710 Project

## 1. Project Overview

GFNet (Graph-Free Vision Transformer) is an advanced vision transformer model designed to handle image recognition tasks. In this project, the main role of GFNet is to distinguish between the two states of cognitive normal (CN) and Alzheimer's disease (AD) from brain MRI scans provided by ADNI (Alzheimer's Disease Neuroimaging Initiative). Through training, the model aims to learn visual patterns and features that distinguish between the two categories to achieve the following goals:

- High Accuracy: The goal of the project is to achieve or exceed a classification accuracy of 0.8, which means that the model is able to correctly identify at least 80% of the test data.
- Robustness: High performance under a variety of conditions and variations, such as images from different stages of the disease or from different scanning devices.
- Generalization ability: The model should be able to handle similar datasets from outside the project, that is, have good generalization ability so that it can be easily adapted to a wider range of medical imaging tasks in the future.

### Advantages of the Model (GFNet)

As a Transformer-based model, GFNet has several obvious advantages when dealing with image data:

- Better feature capture capability: Vision Transformers are able to capture global dependencies and complex patterns more effectively than conventional CNNS, which is especially important for subtle variations common in medical imaging.
- Small amount of prior knowledge requirement: GFNet does not need to rely on domain-specific prior knowledge or complex feature engineering as traditional methods do, making it better adaptive on new or unlabeled medical imaging datasets.
- Efficient information integration capability: The Transformer architecture allows the model to integrate information more effectively when processing large image data, which is particularly critical for parsing and classifying high-resolution medical images.
- Adaptation to complex data distribution: Since medical images often contain very complex biological information and variable representations, the self-attention mechanism of GFNet can provide a powerful way to understand these complexities and improve the accuracy of diagnosis.

In general, GFNet shows better performance and potential than traditional methods when dealing with medical image data, especially high-dimensional and high-complexity, and is suitable as the model for this project.

### Algorithm Implemented

GFNet, a recent breakthrough in vision transformers, presents an advanced architecture for image classification by leveraging the power of Fourier transforms for efficient feature extraction. The GFNet model consists of two key components: feature extraction and classification. The feature extraction part transforms input images into feature maps using global Fourier transformations, allowing the network to capture both local and global patterns efficiently across various scales. Unlike traditional convolutional networks that rely on spatial convolutions, GFNet bypasses these operations by directly operating in the frequency domain, which reduces computational overhead while maintaining accuracy.

## 2. Data Description

### Data Sources

The dataset is provided by Alzheimer's Disease Neuroimaging Initiative (ADNI), with permission to use it from the EECS COMP3710 team at the University of Queensland[^1]. The data has been devided into two floders: train and test, and divided into CN and AD two kinds of data in each folder.

### Data Preprocessing

A comprehensive data preprocessing pipeline was implemented for training a deep learning model using TensorFlow. The pipeline involves several crucial steps designed to prepare image data for effective model training.

1. Each image file path undergoes a series of transformations to ensure it's adequately prepared for the model. This includes reading the image from disk, decoding it into a JPEG format, resizing it to a uniform size (224x224 pixels to match the GFNet model input requirements), and normalizing pixel values to the range [0,1] for better neural network performance. Apart from taht, labels are converted from string format to numerical indices, which are necessary for the model to process the labels effectively. This step also includes creating a unique index for each label which aids in classification tasks.
2. The dataset was construsts by using TensorFlow's tf.data.Dataset API, from the image paths and their corresponding numeric labels, and then enhanced with mapping functions that apply the preprocessing steps concurrently, leveraging TensorFlow’s AUTOTUNE feature to optimize loading and transformation operations. To ensure that the model receives data in manageable sizes and in random order, the dataset is batched and shuffled. This step is crucial for training deep learning models as it helps to reduce memory overhead and introduces randomness in the training process, which can improve the model's generalization ability.
3. The dataset is divided into training and validation sets using a stratified approach based on the labels to ensure that each set represents the overall dataset's distribution accurately. This split helps evaluate the model’s performance during training, providing insight into how well the model might perform on unseen data.

### Data Size & Categorical Distribution

## 3. Model Overview

### Structure of Model (使用的layers, 激活函数, 优化器)

### 模型的设计和实验

### 模型训练的过程和结果，概述训练参数

回调函数的设置 (early stop)

### Challenges and Solutions 遇到的挑战（具体的），解决方案

memory usage - batch size

## 5. 评估结果

### 列出

## 6. Reference

[^1]: “Alzheimer’s Disease Neuroimaging Initiative,” ADNI. <https://adni.loni.usc.edu/>
