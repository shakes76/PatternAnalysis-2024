# Siamese network for Classification of ISIC 2020 Kaggle Challenge data set
## Project Introduction
### Project Summary and Aim
The purpose of this project was to create a Siamese Network that is able to classify skin lesions from the ISIC 2020 Kaggle Challenge data set as either 'normal' or 'melanoma'. The full data set contains 33,126 images of skin lesions  (584 (1.8%) melanoma and 32542 (98.2%) normal).

The aim of this project is to produce a model that is able to achieve an 'accuracy' of 0.8 when the model is used to predict a testing set (set of data that was unseen during training).

### Accuracy  Metric
As noted above the data set is highly unbalanced (1.8% melanoma images and 98.2% normal images). Thus using the standard accuracy score metric to gauge the performance of the model is very misleading. As,for example we could just have our model predict all images as normal thus achieving an 'accuracy' of 98.2% whilst learning nothing about the data and being unable to predict melanomas.

Therefore it was decided that AUR ROC should be used as the metric to gauge the performance of the model. This metric was chosen as it provides a balance between sensitivity (True Positive Rate) and specificity (True Negative Rate) ensuring that both classes are considered (REFERENCE!!!). This metric is very commonly used for imbalanced datasets such as this (REFERENCE!!!). Thus this project will aim to maximise AUR ROC on the testing set. (See ed post #253 for support of this approach (REFERENCE!!!)). Additionally it should be noted that the official kaggle challenge that this dataset originated from used AUR ROC as the metric to determine accuracy of test predictions (REF https://www.kaggle.com/c/siim-isic-melanoma-classification/overview).

## File Structure
### Current Structure
This is the current structure of the project as is when cloned from it
```
PatternAnalysis-2024/recognition/siamese_s46387334/
│
├── readme_figures/
│   ├── readme_image_1.jpg
│   ├── readme_image_2.jpg
│   └── ...
│
├── dataset.py
├── modules.py
├── predict.py
├── train.py
└── README.md
```

### After Data Imports
However to run these scripts it is a requirement that the ISIC 2020 Kaggle Challenge data set is downloaded into this folder. The following section ('Downloading the data') will outline how to do this.
```
PatternAnalysis-2024/recognition/siamese_s46387334/
│
├── readme_figures/
│   ├── example1.jpg
│   ├── example2.jpg
│   └── ...
│
├── dataset.py
├── modules.py
├── predict.py
├── train.py
├── README.md
│
└── data/
    ├── train-metadata.csv
    └── train-image/image/
        ├── ISIC_0015719.jpg
        ├── ISIC_9995166.jpg
        └── ...
```

### After Training and Evaluation
After training and evaluation is run several figures will be saved to this folder to view the outcome of training and testing. The following sections ('Training Details' and 'Evaluation Details') will  run through how to run the training and evaluation and thus produce these figures.
```
PatternAnalysis-2024/recognition/siamese_s46387334/
│
├── readme_figures/
│   ├── example1.jpg
│   ├── example2.jpg
│   └── ...
│
├── dataset.py
├── modules.py
├── predict.py
├── train.py
├── README.md
│
├── data/
│   ├── train-metadata.csv
│   └── train-image/image/
│       ├── ISIC_0015719.jpg
│       ├── ISIC_9995166.jpg
│       └── ...
|
└── produced_figures/
    ├── example1.jpg
    ├── example2.jpg
    └── ...

```



## Environment Setup
### Downloading the data
1. Visit this page to download the data: https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized (REF)

2. Once the data is downloaded - position the data into the folder structure to match the structure listed in the above section


### Python and Package Setup
1. Install Python 3.10.14  (REF)
    - This can be done via the following link https://www.python.org/downloads/release/python-31014/

2. Install the following Python packages (This can be done in a virtual enviroment if desired)
    ```
    pip install pytorch torchvision matplotlib pandas numpy seaborn
    ```
(TODO FINISH - add all moduals nad == for vresions)


## Dataset Details
This section will contain details that pertain mainly to `dataset.py`

### Dataset Introduction
The ISIC 2020 Kaggle Challenge data set is self described as a "Skin Lesion Analysis Towards Melanoma Detection" (REF https://challenge2020.isic-archive.com/). The dataset contains 33,126 dermoscopic images of unique skin lesions from over 2,000 patients. (REF https://challenge2020.isic-archive.com/).

The full original dataset can be sourced from https://challenge2020.isic-archive.com/ (REF). However due to the dataset size - to ensure efficient computation a resized version of the data set was used that was sourced from https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized (REF). From now on the resized data set will be discussed. The images are located in a single folder (train-image/image/) and the classifications of each image are contained within a single csv file (train-metadata.csv) Each of the images from the resized data set are 256 x 256 pixels and a few examples of these images are pictured bellow.

![alt text](readme_figures/sample_data_images.png)

![alt text](readme_figures/sample_data_images_2.png)


### Dataset Train Validation Test Split
The first thing that was done to the data was to split the images up into three sets

- Train Set (80% of all data) 
    - This is the only data that the model will learn from

- Validation Set (10% of all data)
    - To be used for validation purposes while training (or be used for hyper parameter tuning)
    - To gauge how the current model is preforming on unseen data

- Test Set (10% of all data)
    - Will be used to evaluate the performance of the final model

The Validation and testing sets were kept relatively small to maximise performance of the model by giving it a larger set of data to train on. Due to the large size of the data set the validation and testing sets are still large enough to get a good idea of the population.

### Dataset Oversampling and Augmentation
Due to the large class imbalance mentioned above it is preferred if a method is used to ensure that the model is trained on balanced data (REF).To do this we use a two step approach - oversampling the minority class, then data augmentation. It should be noted this class balancing was only done for the training set.

For oversampling - the minority class (melanoma) was oversampled until both classes had an equal number of data points. Then to ensure that this oversampling does not lead to overfitting (i.e. model could memorize the duplicated instances without learning meaningful patterns) - we apply data augmentation to all the sample (we apply the augmentation to both classes to ensure the model does not just learn that certain augmentations indicate a certain class).

Augmentation methods used are as follows (see the following documentation for details https://pytorch.org/vision/main/transforms.html)  (REF):
- RandomRotation (Randomly rotate the image)
- RandomHorizontalFlip (Randomly horizontally flip the image)
- RandomVerticalFlip (Randomly vertically flip the image)
- ColorJitter (Randomly adjust the brightness, contrast, saturation and hue of the image)

> ### How to use dataset.py
> - The full data set (images and labels) and be accessed via the get_isic2020_data() function in dataset.py
> - The train (with oversampling), validation, test split data loaders (with augmentation applied) can be accessed accessed via the get_isic2020_data_loaders() function in dataset.py


## Siamese Network Details
This section will contain details that pertain mainly to `modules.py`

### Siamese Network Overview


### Embedding Model Architecture

### Classifier Model Architecture

### Embedding Model Loss function (Triplet Loss)

### Classifier Loss function


> ### How to use modules.py
> - The full model (embedding + classifier model) can be accessed via
> - The config used for the model can be accessed via 


## Training Details
This section will contain details that pertain mainly to `train.py`


## Evaluation Details
This section will contain details that pertain mainly to `predict.py`





## Results Summary



## Future Work and Improvments



## References







## Results Summary



## Future Work and Improvments



## References

