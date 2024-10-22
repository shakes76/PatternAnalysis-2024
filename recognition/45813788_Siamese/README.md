# Siamese based network for ISIC Dataset using contrasitive Loss

## Problem Description
The goal of this project was to develop a siamese network that would perform similarity checks on skin lesions and then extract the embeddings from this network to classify the images into one of the following categories

- Benign (skin lesion is not melanoma)
- Malignant (Melanoma Detected)

## Data
The Dataset provided for this project was the ISIC 2020 Kaggle Dataset. for efficiency a scaled down version of this dataset containing 256x256 images was used and can be found [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data). One of the primary challenges of this task was the high imbalance of classes, with only ~600 malignant images and ~30,000 benign images typical metrics such accuracy would not be suitable for this task. Thus the solution involved using the metric outlined by Kaggle being the receriver operate under the Receiver Operator Partial Under the Area Curve (roaruc). In additon classification reports were used which would contain metrics such as F1 score, Recall and Precision.

To handle imbalanced datasets there are several methods that can be used to combat this being: 
- data augmentation 
- under-sampling of the larger class set 
- using a sampler for each batch to ensure even amounts of both classes (effectively resampling the smaller class several times). 

The methods above in combination with metric learning methods such as contrastive loss which generates similar and dissimlar pairs helps to effectively combat the class issue. The aforementioned pairs will result in a large amount of training data. As contrastive loss is being used the purpose is to minimize distance between similar observations and maximize distance for dissimlar pairings.


### Preprocessing



# References

[1]
Z. You et al., [“Vocal Cord Leukoplakia Classification Using Siamese Network Under Small Samples of White Light Endoscopy Images,”](https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/abs/10.1002/ohn.591) Otolaryngology-head and neck surgery, vol. 170, no. 4, pp. 1099–1108, 2024, doi: 10.1002/ohn.591.

