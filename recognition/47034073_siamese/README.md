# Detecting Malignant Lesions From Highly Imbalanced Data Using Triplet Loss

## Introduction 
The 2020 ISIC malignant lesion detection challenge on Kaggle was the task of correctly classifying images of skin lesions as malignant or benign. The dataset is highly imbalanced with approximately 98% of observations being benign and only 2% being malignant. This meant that using naive metrics such as accuracy was undesirable as a predictor which simply always predicted malignant lesions would obtain 98% accuracy. This is why the challenge was to achieve a high AUC score.

There are a few techniques for dealing with highly imbalanced data such re-sampling and heavy augmentation. Metric learning is one such method which alleviates the issue by generating a large amount of training data in the form of observation combinations. Because of the combinatorial explosion of pairings we need worry less about the class imbalance. Metric learning is the method of directly learning feature embeddings which maximimise or minimize some distance between observations. For our problem we wish to minimize the distance between observations from the same class and maximize the distance between osbervations from different classes. 

## The Training Algorithm and Triplet Loss

