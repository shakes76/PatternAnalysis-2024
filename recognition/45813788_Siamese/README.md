# Siamese based network for ISIC Dataset using contrasitive Loss

## Problem Description
The goal of this project was to develop a siamese network that would perform similarity checks on skin lesions and then extract the embeddings from this network to classify the images into one of the following categories

- `Benign` (skin lesion is not melanoma) with a value of `0`
- `Malignant` (Melanoma Detected) with a value of `1`

## Data
The Dataset provided for this project was the ISIC 2020 Kaggle Dataset. for efficiency a scaled down version of this dataset containing `256x256` images was used and can be found [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data)[1]. One of the primary challenges of this task was the high imbalance of classes, with only ~600 malignant images and ~30,000 benign images typical metrics such accuracy would not be suitable for this task. Thus the solution involved using the metric outlined by Kaggle being the receriver operate under the Receiver Operator Partial Under the Area Curve (roaruc). In additon classification reports were used which would contain metrics such as F1 score, Recall and Precision.

## Preprocessing
To handle the imbalanced dataset the following methods were used:
- importing the excel and dropping the `unamed` and `patient id` columns (We will use ISIC as it is unique and images named by this)
- data augmentation
- normalisd pixel values between `0` and `1`
- using a sampler for each batch to ensure even amounts of both classes in training (effectively resampling the smaller class several times).
- validation and testing sets had shuffing applied


Once this was complete the dataset was split into a `75/15/10` split between training/validation/testing. In addition stratification was used based on class so that subsets have similar proprotions to further balance classes.

As the contrastive loss in pytorch metric learning was used there was no need to to create pairs manually as the method creates these paris internally.Because of this no additional data processing outside of what was done above needed to be done. It should be mentioned that these pairs also help to combat imbalance as the combinations of pairings increases the data the model has to work with.

## Architecture
The final model architecture is based on the [“Vocal Cord Leukoplakia Classification Using Siamese Network Under Small Samples of White Light Endoscopy Images,”](https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/abs/10.1002/ohn.591) by You et al and can be seen below:
![Siamese Model Architecture](./images/MODELARCHITECTURE.jpg) 


Figure 1: Siamese Netowrk Architecure [2]

Some modifications have been made such as using Binary Cross Entropy (BCE) Loss using logits with a sigmoid in training and testing whenever probabilities needed to be obtained (BCE with logits does sigmoid internally so you don't add that layer to the model architecture itself). this decision was made as the problem space is binary compared to the multiclassification problem in the report, that being said the same could be done with Cross Entropy using 2 classes. In additon to this the siamese backbone was implemented using `resnet50` this approach was taken due to the realtive sucess that was attained in  “Identifying Partial Mouse Brain Microscopy Images from the Allen Reference Atlas Using a Contrastively Learned Semantic Space,” [3].

## Loss Functions
The key feature this network proposed was in how it computes the loss for the network, instead of treating the classification head and embedded network seperately the network instead computes the loss as follows:


**$$L = L_{BCE} + L_{CL}$$**

where:

**$$L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$** 

[4]


contrasitive loss using `Lp` distance (Euclidean as p=2):

**$$ L_{CL} = [d_p - m_{pos}] + [m_{neg} - d_n] $$**


[5]

Thus it effectively improves the embeddings by minimising distnace whilst also having guidance from a classification perspective, this feature was something I did not see in other papers and lead to good perfomance as can be seen below.

## Model Perfomance
### Training and validation
the model was trained for 15 epochs and lead to the following results:
![ModelLoss](./images/newmodloss.png) 
![ModelAccuracy](./images/newmodacc.png) 
![trainAuroc](./images/newmodauroc.png)

Observing above it can be seen that there was some overfitting past epoch 5. as per kaggle the AUROC was used as the basis of determining the best performing model to use for testing which was epoch 5, observing it's embeddings and classification scores:
![Embeddings](./images/epoch5.jpg)

Classification on the validation set of epoch 5:

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| 0.0      | 0.99      | 0.86   | 0.92     | 4881    |
| 1.0      | 0.07      | 0.58   | 0.12     | 88      |
| **Accuracy**   |           |        | 0.85     | 4969    |
| **Macro Avg**  | 0.53      | 0.72   | 0.52     | 4969    |
| **Weighted Avg**| 0.97     | 0.85   | 0.91     | 4969    |

looking at this precision table it can be seen that the model performs well for class 1 overall with a high F1 score but still struggles to to identify class 1 with a low precision but when observing recall it becomes more apparent that this is due to the model predicting a large amount of false positives.

### Testing
![trainAuroc](./images/newmodtestauroc.png)
![testembed](./images/epochTest.jpg)

### Testing Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 0.84   | 0.91     | 3255    |
| 1     | 0.08      | 0.76   | 0.14     | 58      |
| **Accuracy**   |           |        | 0.84     | 3313    |
| **Macro Avg**  | 0.54      | 0.80   | 0.53     | 3313    |
| **Weighted Avg**| 0.98     | 0.84   | 0.90     | 3313    |

### Overall Test Accuracy: **81.73%**

### Confusion Matrix:

|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| Actual 0   | 2732        | 523         |
| Actual 1   | 14          | 44          |

Observing the results above we see the same issue as in training this being that the poor f1 score is attributed to false positives. That being said the recall for malignant cases was 76% which is actually very good consideirng the major class imbalance. the confusion matrix also alludes to better results showing how the majoirty of malignant samples were correctly classified. 

Observing the PCA and tSNE embeddings for both the training and test sets we can see that some clustering of malignant poitns is occuring but no evident seperation, this alludes to potentially requirng a deeper network and further hyper parameter tuning but the AUROC scores and accuracy say otherwise. thus it would be worthwhile to first test the network again on a more balanced dataset and see if it performs well there. For those that want to continue using the dataset in this problem it is reccomended to try a deeper resnet to see if more meaningful feature extractions can be made. If hardware and time permits reduce the learning rate and increase epochs to see if the resnet architecture can create more distinctions between the classes in the embedding space, this would in turn lead to a stronger classifier.


## Dependencies
All work was done using conda to manage virtual environments and package. the environment can be intialised by doing the following
```
conda env create –f environment.yml
```
for those not using conda the primary packages were as follows:
```
  - python=3.11.10
  - pytorch=2.4.1=py3.11_cuda11.8_cudnn9_0
  - torchvision==0.19.1
  - pytorch-metric-learning=2.6.0
  - matplotlib=3.9.2
  - pillow=10.4.0
```
## Training and Testing
All training and testing of the model is done by one script: `driver.py`. to run the script the following can be specified:
```
python driver.py -m {train/test/both} -p {True/False}
```
where:
- `m` specifies the mode between training testing or both
- `p` specifies if you want the training plots

NOTE: You must train first as testing requiresa saved model to use.

##Repository Structure
The file tree must be as follows
```
root
├── dataset
│   ├── train-image
│   │   ├── ISIC--.jpg
│   │   ├── ...
│   │   └── ISIC--.jpg
│   └── train-metadata.csv
│       
└── images
│   │   ├── epoch1.jpg
│   │   ├── ...
│   │   └── epochTest.jpg
└── models
      └── siamese_best.pth
```
NOTE: the python scripts create the models and images directory internally but it is safer to make them before hand incase there are conflicts with permissions.

Hyperparameters can be edited inside the file named `hyper.py`, currently only batch_size, learning rate and epochs were modified (m size for sampler was left as batch_size/2)

## Additional Remarks
- when attempting earlier iterations with split classifier and siamese i was getting significantly worse results. i managed to reach an auroc score of 0.8 but my actual accuracy was quite low at 37%. that being said that run did not miss classify any malignant images but had more false positives then false negatives the truth table for this run is as follows:


|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| Actual 0   | 1173        | 2082         |
| Actual 1   | 0          | 58          |



- it's important to augment both classes and not just the minority class. when i did this the performance scores for training and validation were good but come testing the model had 98% accuracy by having correctly classfied all benign images and no malignant images correctly. the embeddings looked good to with great distinctions but what had happened here is the model had just learnt how to distinguish the augmentations.
- For all testing purposes i chose not to augment images as i wanted it to perform on unseen images that you would expect in this format. there's actually quite a bit of debate as to whether to test on augmented images so its worth trying to see if the model performs better if all sets have augmentation as this may show some memorization of augmentation vs actual distinctions.
- In additon i intially had coluor jitter for my augmentation but for this dataset i felt that it didnt make sense since melanoma that is cancerous would have changes in its color so it didnt make sense to colour jitter to me.

# References

[1]
Nischay Dhankhar, “ISIC 2020 JPG 256x256 RESIZED,” Kaggle.com, 2020. https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data

[2]
Z. You et al., [“Vocal Cord Leukoplakia Classification Using Siamese Network Under Small Samples of White Light Endoscopy Images,”](https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/abs/10.1002/ohn.591) Otolaryngology-head and neck surgery, vol. 170, no. 4, pp. 1099–1108, 2024, doi: 10.1002/ohn.591.

[3]
J. Antanavicius, R. Leiras, and R. Selvan, [“Identifying Partial Mouse Brain Microscopy Images from the Allen Reference Atlas Using a Contrastively Learned Semantic Space,”]() in Biomedical Image Registration, Cham: Springer International Publishing, pp. 166–176. doi: 10.1007/978-3-031-11203-4_18.

[4]
N. J. Nathaniel, "Understanding PyTorch loss functions: The maths and algorithms (Part 2)," Towards Data Science, Aug. 10, 2021. [Online]. Available: https://towardsdatascience.com/understanding-pytorch-loss-functions-the-maths-and-algorithms-part-2-104f19346425.

[5] K. Musgrave, “Losses - PyTorch Metric Learning,” Github.io, 2024. https://kevinmusgrave.github.io/pytorch-metric-learning/losses/.

