# Detecting Malignant Lesions From Highly Imbalanced Data Using Triplet Loss

## Usage
### Data Preparation
### Downloading The Trained Model
### Training
### Example Prediction

## Problem Overview 
The 2020 ISIC malignant lesion detection challenge on Kaggle was the task of correctly classifying images of skin lesions as malignant or benign. The dataset is highly imbalanced with approximately 98% of observations being benign and only 2% being malignant. This meant that using naive metrics such as accuracy was undesirable as a predictor which simply always predicted malignant lesions would obtain 98% accuracy. This is why the challenge was to achieve a high AUC score.

There are a few techniques for dealing with highly imbalanced data such re-sampling and heavy augmentation. Metric learning is one such method which alleviates the issue by generating a large amount of training data in the form of observation combinations. Because of the combinatorial explosion of pairings we need worry less about the class imbalance. Metric learning is the method of directly learning feature embeddings which maximimise or minimize some distance between observations. For our problem we wish to minimize the distance between observations from the same class and maximize the distance between osbervations from different classes. 

## Data Preparation
Patient meta-data was not used for simplicity as other solutions had found that they were not very essential performance <citation>. Although it would be simple to incorporate this data by simply concatenating it to the outputted feature embeddings and apply appropriate normalization.

Data was split into a train, validation and test set. 64% of data was used for training, 16% was used for validation and 20% was used for testing. Stratified sampling was used to ensure that the ground truth distributions were well represented. This was a fairly important consideration as there was not much data in the minority class, stratification ensured all sets had adequate examples.

Images were all processed pre-training by first downsampling them to a 256x256 resolution and then taking a center crop of 224x224 pixels. This was done to speed up training and and make the data compatible with ResNet50 embedding model architecture.

## The Training Algorithm and Triplet Loss
A triamese network was used with the triplet loss[^1], and extension of the siamese network <source>, 

## Results

## Discussion

## Conclusion

## References
[^1]: Schroff, F., Kalenichenko, D., & Philbin, J. (2015, June). FaceNet: A unified embedding for face recognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/cvpr.2015.7298682