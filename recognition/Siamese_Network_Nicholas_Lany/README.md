# Siamese Network Classifier for ISIC 2020 Dataset

This project implements a **Siamese neural network** to classify skin lesion images from the **ISIC 2020 Kaggle Challenge dataset**, aiming to differentiate between normal skin and melanoma.

## Description

Melanoma is a serious form of skin cancer, and early detection is critical for effective treatment. In this project, we create a **Siamese network-based classifier** to distinguish between images of normal skin and melanoma using the ISIC 2020 dataset. The goal of this model is to achieve an accuracy of around **0.8** on the test set. The model addresses the problem of binary classification for medical images, specifically to detect malignant melanoma in dermoscopic images.

## How It Works

The algorithm utilizes a Siamese network, which takes pairs of images as input and learns to predict whether the images are of the same class (benign or malignant). The model employs a CNN as a feature extractor, followed by a contrastive loss function to train the network. By learning the similarity between image pairs, the model can classify unseen data effectively. Furthermore, our classification approach differs slightly, as we used the trained CNN that is a byproduct of our Siamese network to classify the data.

## Inputs

The inputs to the siamese network are 256x256 images of benign and malignant melanoma. Here is an example of this input:
![Example Input](images/example_input.png)

## Outputs

The output of this network is a value that classifies whether an image shows benign or malignant melanoma. For example if the output is 1, the image shows malignant melanoma and vice-versa for benign cases.

## Loss

As the contrastive loss is minimised, the siamese network gets better at distinguishing between pairs that are alike and pairs that are different. This model did take a while to train and minimise this loss function, but as seen below on a run with only 25 epochs, the loss does decrease over time. It was also found that marginal decreases in the loss would still dramatically improve the CNN that is used to classify the images.

## Dependencies

-   Python 3.12
-   torch==2.0.1
-   torchvision==0.15.2
-   pandas==2.1.1
-   PIL==10.0.0
-   kagglehub==0.3.0

Ensure that the required libraries are installed using:

```bash
pip install torch torchvision pandas kagglehub pillow
```

## Preprocessing

Preprocessing includes resizing the images to 256x256 pixels and converting them to tensors. Malignant images undergo data augmentation with random rotations, flips, and color jittering to generate five additional samples per malignant image. This helps the model generalize better and counteract the imbalance of benign vs. malignant cases in the dataset. The augmentation is only applied to the malignant cases.

## Dataset

The data was used to create pairs of two images with their associated similarity (if they are both the same type, they will have a label of 0, and if they are different they have a label of 1). Since pairs were created, the dataset could be used to create different combinations, and for this reason, 50,000 pairs could be generated with ~33,000 images. This data was used to train the model, while the test data would be tested on 1000 random samples of the dataset that were not previously used to train the model

## Reproducibility

These results are reproducible by running the `python train.py`, which saves the model to the working directory as 'model.pth'. By running `python predict.py` in the same directory, the accuracy of the saved model will be evaluated.

## Results

The results show that the CNN that was trained by the siamese network can achieve an accuracy of xx%. This achieves our goal of >80% accuracy in classifying images of melanoma.
