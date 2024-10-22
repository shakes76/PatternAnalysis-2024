# Siamese network for Classification of ISIC 2020 Kaggle Challenge data set
## Project Introduction
### Project Summary and Aim
The purpose of this project was to create a Siamese Network that is able to classify skin lesions from the ISIC 2020 Kaggle Challenge data set as either 'normal' or 'melanoma'. The full data set contains 33,126 images of skin lesions  (584 (1.8%) melanoma and 32542 (98.2%) normal) and will have to be split up into train, validation and test sets as apart of the training and testing precedure. 

The aim of this project is to produce a model that is able to achive an 'accuracy' of 0.8 when the model is used to predict a testing set (set of data that was unseen during training).

### Accuracy  Metric
As noted above the data set is highly unbalanced (1.8% melanoma images and 98.2% normal images). Thus using the standary accuracy metric to gauge the preformance of the model is very missleading. As, for example we could just have our model predict all images as normal thus acchiving an 'accuracy' of 98.2% whilst learning nothing about the data and being unable to predict melanomas.

Therefore it was decided that AUR ROC should be used as the metric to gauge the preformance of the model. This metric was chosen as it provides a balance between sensitivity (True Positive Rate) and specificity (True Negative Rate)ensuring that both classes are considered, and therefore evaluates the model's ability to correctly identify both minority and majority classes (REFERENCE!!!). This metric is very commonly used for imbalanced datasets such as this (REFERENCE!!!). Thus this project will aim to maximise AUR ROC on the testing set. (See ed post #253 for support of this approch (REFERENCE!!!))

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

### After Training and Evaulation
After training and evaluation is run several figures will be saved to this folder to view the outcome of training and testing. The folling sections ('Training Details' and 'Evaluation Details') will  run through how to run the training and evaulation and thus produce these figures.
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



## Enviroment Setup
### Downloading the data



### Package Installations






## Dataset Details





## Siamese Network Details



## Training Details




## Evaluation Details





## Results Summary



## Future Work and Improvments



## References







## Results Summary



## Future Work and Improvments



## References

