# Metric Learning for Melanoma Classification

The contents of this repository address the ISIC2020 Challenge of classifying benign and malignant skin markings. 

## Siamese Networks and Metric Learning

This repository utilises a type of neural network known as a Siamese network. Instead of directly classifying images, Siamese networks learn an embeddings network which maps the input data to a manifold, with the intention of making such a manifold which maximises the separation between two classes. The foundations of this type of metric learning are in facial recognition, where there is a large amount of general data but very little data about specific people. Where traditional learning can degenerate due to this limited information, Siamese networks can instead thrive. As they effectively learn to quantify the similarity between two images, only a single image of the positive class is needed to perform classification.
This task was initially completed on the MNIST dataset, and the embedding visualisations provide a strong argument for using Siamese Networks. The below image shows the output of the embeddings layer after training the network on the MNIST dataset. The manifold it creates performs exceptionally well at distinguishing the distinct classes from one another, making subsequent classification trivial.

## Data Preprocessing
The ISIC2020 dataset is challenging to work with for a number of reasons. Firstly, it is heavily biased towards the negative class, with about 98% of the dataset being benign cases. This can make it difficult for the classifier to learn meaningful distinctions between benign and malignant cases. To account for this, the positive class of the dataset was oversampled to the be the same proportion as the negative class.
In addition to having an unbalanced dataset, there were several cases of data leakage. In a few instances, multiple images were taken from the same skin lesion. This can be problematic as test sets are usually assumed to be comprised of unseen data, however having some replicas of the training data can give a false representation of the true testing performance.
Further to this, the image size throughout the dataset was not homogeneous, with some images being up to 2000x4000. Given the length of this that this resolution of dataset would require, a downsize 256x256 replica of the dataset was instead chosen.
Because of how few positive images there were, heavy data augmentation had to be applied. Random flips and rotations were implemented, as it makes physical sense that changing how the picture of a lesion is taken should not cause the model to break. Other augmentations, such as random brightness, noise and contrast were experimented with but ultimately deemed not beneficial.
A standard 80/20 split is used for training and testing. However, when doing so special care must be made to stratify this split, as a training dataset with no positive classes could not perform. This also ensures the testing results are representative of the training that occurred. After this stratified split, the training set was further broken down into 70/10 for training and validation.
## Siamese Network Implementation
The ResNet50 model was used as the feature extractor for the melanoma dataset, with several fully connected layers being added after the ResNet to obtain meaningful embeddings. Triplet loss, a common loss function for metric learning, was used as the loss function for the similarity network. This loss function works by comparing a positive and a negative image to a reference anchor, and then trying to minimise the distance to the positive class while maximising the distance to the negative class. Mathematically, it is shown as

$$L(x_a,x_p,x_n )=max⁡(0,|x_a-x_p |^2-|x_a-x_n |^2+m)$$

In this equation, m is the margin of the loss function, and represents how far apart the classes are being driven.
Although Siamese networks can be done with two distinct networks, it is more common to use a single network and a miner. When using triplet loss, the miner is responsible for generating triplets of images in a way that forces the network to learn. These are often referred to as semihard triplets and are subject to the following constraint:

$$|x_a-x_p |<|x_a-x_n |<|x_a-x_p |+m$$

In this architecture, a semi-hard triplet loss was used, with a margin set to 0.4.
The other hyperparameters chosen for the metric learning are as follows:
- Batch size of 128
- Image size of 256x256
- Learning rate of 0.001
- Epochs of 20
- Optimiser of Adam

Once class separation had been attempted, the layers of the embeddings network were frozen, and a new fully connected network was created from the output embeddings. Given that the intention of Siamese Networks is to obtain meaningful embeddings, the classification network did not need to be complex. Three dense layers were chosen (downsizing from 32 to 16 to 4), with the output layer being fed through a sigmoid activation to obtain probabilistic outcomes. 50 epochs were used for the classifier, using the Adam optimiser and binary cross entropy as the loss function. 

## Results

## Dependencies
This network relies on the use of tensorflow_addons, which is deprecated as of mid-2024. For compatibility reasons, this model therefore uses an older version of Python (3.11.1) and numpy/tensorflow. To install, you must first have Python 3.11.1, or create a virtual environment that supports it. All relevant packages can then be installed using `pip install -r requirements.txt`

##d Disclaimer
The commit log admittedly is inconsistent. This is primarily because trials were run on different branches with more than questionable commit history. For a clean history, these incremental changes were copied over to the main branch. 
