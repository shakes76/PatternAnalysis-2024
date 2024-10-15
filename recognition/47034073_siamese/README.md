# Detecting Malignant Lesions From Highly Imbalanced Data Using Triplet Loss

## Usage
### Data Preparation
### Downloading The Trained Model
### Training
### Example Prediction

## Problem Overview 
The 2020 Kaggle ISIC malignant lesion detection challenge was to correctly classify images of skin lesions as malignant or benign. The dataset is highly imbalanced with approximately 98% of observations being benign and only 2% being malignant. This meant that using naive metrics such as accuracy was undesirable as a predictor which simply always predicted malignant lesions would obtain a score of 98%. Therefore the challenge was to achieve a high reciever-operator-curve area-under-the-curve (ROC-AUC) score.

There are a few techniques for dealing with highly imbalanced data such re-sampling and heavy augmentation. Metric learning is one such method which alleviates the issue by generating a large amount of training data in the form of observation combinations. Because of the combinatorial explosion of pairings we need worry less about the class imbalance. Metric learning is the method of directly learning feature embeddings which maximimise or minimize some distance between observations. For our problem we wish to minimize the distance between observations from the same class and maximize the distance between observations from different classes. 

## The Training Algorithm and Triplet Loss
A triamese network was used with the triplet loss [[2]](#2), an extension of the siamese network popularised for one-shot image recognition [[1]](#1). For this training framework we first compute feature embeddings outputted by the second last layer of a ResNet50 before softmax is applied. The ResNet architecture can be seen in Figure [num](). Pre-trained ImageNet weights were not used, as it was thought that features trained on natural objects would not transfer well to lesion images. Pre-trained network weights have been shown to be inflexible in some cases [[]](). The ResNet50 produced features are then passed through an additional embedding head which creates a 256-dimensional latent embedding vectors. The final embedding is then normalized such that has unit $L_2$ norm. Let the final normalized embedding be given by $f(x)$ where $f$ is the embedding mapping and $x$ is lesion image. The architecture of the embedding head can be seen in Figure.

Once embeddings are produced they are compared using triplet loss. The triplet loss requires the comparison of three training examples at a time, an anchor example, a positive example from the same class as the anchor and a negative example from a different class to the anchor. Let $l(x^a, x^p, x^n)$ be the loss for an single triplet with anchor $x^a$, positive $x^p$, and negative $x^n$ lesion images. Then $l$ is given by:

$$l(x^a, x^p, x^n) = \mathrm{ReLU}\left(\Vert f(x^a) - f(x^p) \Vert_2^2 - \Vert f(x^a) - f(x^n) \Vert_2^2 + m \right)$$

Where $m$ is some hyperparameter for the desired margin. Note that $\mathrm{ReLU}$ is nothing more than the maximum of its argument and 0, this meant that 0 is the smallest loss possible. The left term of the loss encourages pairs from the same class to be close to each other in embedding space. The right term encourages pairs from different class to be far apart in the embedding space. The minimum loss is acquired when the positive example is closer than the negative example and is closer by margin $m$.

One of the most import parts to triplet loss performance is the mining of triplets. This refers to the process of finding difficult triplets to train with, as triplets with zero loss have no learning signal. The most popular mining technique is semi-hard mining as introduced in [[]](). This technique chooses triplets such that the positive example is closer to the anchor than the negative sample but the margin is still violated. These triplets are easier to improve than when the negative example is closer than the postive one, but still provide a learning signal. This strategy is often used to improve stability of learning. However, it was found empirically for this problem that the all mining strategy worked the best. This strategy selects all triplets which violate the margin. Triplets are sampled online from each minibatch.

After training the embedding network we use embeddings as an input to a separate classifier. 

The embedding network was trained with Pytorch using the Adam optimizer with a learning rate of $1 \times 10^{-5}$ and weight decay of $1 \times 10^{-6}$. All other Adam hyperparameters were set to the Pytorch defaults. the network was trained on a single RTX 3060 GPU.

For this project three classifiers were tried using sci-kit learn implementations, a K nearest neighbours (KNN), a support vector machine (SVM) and a multilayer perceptron (MLP). The KNN used a custom uniform voting from the nearest neighbor and all neighbours that were closer than $m$. The SVM used the default sci-kit learn hyperparameters, most notably a $C$ value of 1 and the Radial Basis Function kernel. The neural network architecture can be seen in Figure [num](), the Adam optimizer was used with a learning rate of 0.0001 and a momentum value of 0.9, other Adam hyperparameters were set to the sci-kit learn defaults.

## Data Preparation
Patient meta-data was not used for simplicity as other solutions had found that they were not very essential performance [[]](). Although it would be simple to incorporate this data by concatenating it to the outputted feature embeddings and apply appropriate normalization.

Data was split into a train, validation and test set. 64% of data was used for training, 16% was used for validation and 20% was used for testing. Stratified sampling was used to ensure that the ground truth distribution was well represented in each set. This was a fairly important consideration as there was not much data in the minority class, stratification ensured all sets had adequate examples from this class.

Images were all processed prior to training by first downsampling them to a 256x256 resolution and then taking a center crop of 224x224 pixels. This was done to speed up training and and make the data compatible with the embedding model architecture. 

Image pixel intensities were rescaled to be in the range [0, 1] by dividing through by the RGB maximum of 255. Images were also augmented in an online-fashion during training with a random horizontal flip with probability 0.5 and then a random vertical flip with probability 0.5. Colour augmentation was not used as the true colour was likely to be a very important feature for classification of lesions [[]]().

Equal class sampling during embedding network training was tried to help alleviate class imbalance. This is a sampling technique where each minibatch is formed by exactly half of the observations from each class. This was found empirically to perform worse than standard random minibatch sampling which was more representative the ground truth distribution. However, for classifier training, precision and recall for the malignant class would perform poorly using the whole training set. Instead, for this stage the majority class was undersampled to match the size of the malignant class, that was 374 observations from each class.

## Training Details

## Results
The training loss curve can be seen in Figure [num]().

## Discussion

## Conclusion

## References
<a id="1">[1]</a> Koch, G.R. (2015). Siamese Neural Networks for One-Shot Image Recognition.

<a id="2">[2]</a> Schroff, F., Kalenichenko, D., & Philbin, J. (2015, June). FaceNet: A unified embedding for face recognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/cvpr.2015.7298682