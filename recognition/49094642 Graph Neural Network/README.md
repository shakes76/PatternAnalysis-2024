# Graph Neural Network for Classification on the Facebook Large Page-Page Network Dataset

Author:Zhe Wu
student ID:49094642

## Project Overview

## Table of Contents
- [Environment Dependencies](#environment-dependencies)
- [Inputs](#inputs)
- [Model Usage](#model-usage)
- [Outputs](#outputs)
- [References](#references)



## Environment Dependencies
The project requires the installation of the following software or packages:
- Python 3.12.4
- Pytorch 2.4.1
- Cuda 11.8 
- Numpy 1.26.4
- scikit-learn 1.5.1
- Pandas 2.2.2
- Torch Geometric 2.6.1
- UMAP-learn
- Matplotlib

## Inputs

## Model Usage

## Outputs
The dataset is divided into train set, validation set and test set according to 80%, 10% and 10%. And the learning rate is set to 0.005. After 400 epochs, the best train accuracy is 0.9409 and the test accuracy is 0.9206. The accuracy and loss values of the train set and test set are as follows:

图片

The visualization curves of accuracy and loss value corresponding to the training set and test set of the entire training process are as follows：

图片
图片

From the loss curve, we can see that both curves are high at the beginning, and then gradually decrease. After about 50 epochs, the loss value stabilizes and finally approaches 0.3. Although the training loss is significantly lower than the test loss, this indicates that the model performs better on the training set, but there may also be some overfitting trends.
From the accuracy curve, we can see that both curves rise rapidly at the beginning, then gradually stabilize, and finally fluctuate between 0.85-0.9. 
We visualize the output results and use UMAP to reduce the dimensionality of the high-dimensional feature vector to a two-dimensional projection:

图片



## References
- [1] Distill. 'A Gentle Introduction to Graph Neural Networks', Accessed 10/27. https://distill.pub/2021/gnn-intro/
- [2] Boldenow, Brad. 2018. 'Simple Network Analysis of Facebook Data', Accessed 10.26. https://www.kaggle.com/code/boldy717/simple-network-analysis-of-facebook-data



