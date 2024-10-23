# Siamese Network Classifier for Melanoma Skin Cancer Detection and Classification
# 1. Project Overview
This project implements a Siamese-based classification network to classify the ISIC 2020 Kaggle Challenge dataset. The packages includes data pre-processing and organisation, network design with adjustable hyperparameters, as well as training and evaluation scripts. The target accuracy for classification is 80% on the evaluation dataset.
## 1.1 Dataset Overview
The dataset employed in this project is the ISIC 2020 Kaggle Challenge Dataset. A resized version of the dataset was used, with source data in the form of JPG images with dimensions 256x256 pixels (RGB). Two classes are present in the dataset - Benign and Malignant. A CSV file is also present, containing image IDs, patient IDs, as well as class labels. Each patient had multiple different images at various skin positions and angles present within the dataset.
The dataset could be accessed through [this link](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized).
# 2. Environment Setup
This package has only been tested on windows devices, though theoretically should function similarly in other operating systems. Seperate installation instructions are present below for both conda and non-conda based installations.
## 2.1 Conda Environment Setup
1. Install [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)
2. Within the Miniconda terminal, navigate to the directory of this project, and execute the following commands to create and activate the environment
```
conda env create -f environment.yml
conda activate Siamese
```
This will setup the conda environment will all necessary dependencies automatically installed.
## 2.2 Non-Conda Environment Setup
Alternatively, the required packages could be separately installed if the usage of conda is not desired. The following core dependencies must be installed:
* python=3.9.20
* pytorch=2.1.1+cu11.8
* torchvision==0.16.1
* torchaudio==2.1.1
* cuda-runtime=11.8.0
* cuda-toolkit=11.8.0
* scikit-learn=1.5.1
* scipy=1.10.1
* seaborn=0.13.2
* matplotlib=3.9.2
* numpy=1.24.3
* scipy=1.10.1
# 3. Model and Training Framework
Run `python3 train.py` to run the training script.
## 3.1 Data Pre-processing
The dataset used only contains two classes, and there exists a significant imbalance in the data count between the two classes. There is significantly more benign images compared to malignant images, which can be detrimental to the effectiveness of model training. Additionally, all data is placed in a single directory with the data labels stored separately, requiring additional processing to match images to their labels. As such, the dataset is pre-processed prior to network training.
The `dataset.py` script covers all pre-processing of data, which consists of the following steps:
1. Data Classification and Separation
Runs for the first time data is imported only. Each image is matched by their isic_id to the csv file, and separated into different directories based on their class label. This standardises the way the data is stored, allowing for ease of data retrieval and processing.
2. Data Class Balancing
Imbalance in data classes is addressed by oversampling the minority class (malignant). Random samples of the minority class is selected with repetition until the balance between classes reaches the sampling ratio (`OversampleRatio` in `config.yaml`). Balancing of classes is necessary to prevent biases in model predictions while also guaranteeing sufficient data is present for all classes for a well-trained model.
3. Train-Test Data Split
Balanced dataset is split based on a splitting ratio (`TrainTestRario` in `config.yaml`) into training and validation datasets. The data is well mixed prior to splitting to ensure a mix of both classes is present in both training and validation. The splitting method used also ensures complete separation between the two datasets, preventing the repetitive presence of data impacting network performance.
4. Training Data Augmentation
Training data is augmented in various ways to encourage generalisation in training for better performance, as well as diversifying the duplicated copies of the minority class. Validation data is not augmented to maintain data integrity and allow the direct deployment of data for immediate results in practice. Augmentations used are randomised reflections in both axis, as well as randomised modifications in HSV.
To run the script, download the dataset and extract into a folder in the project workspace. Modify the `DatasetImageDir`, `LabelCSV`, and `OutputDir` parameters in the `config.yaml` file to set the data directory, CSV file path, as well as the name of the output directory to be generated.
Following this, run `python3 dataset.py` to run the pre-processing script.
## 3.2 Network Architecture and Training
The Siamese network architecture consists of a pair of neural networks with shared design, weights, and biases. These networks are use to identify dissimilarities between pairs of datapoints. The type of neural network used in a Siamese network can vary greatly, and is selected differently depending on the form of data and requires output.
### 3.2.1 Network Explanation
For the neural network pair, the ResNet50 network is used. This network was selected for its ability to extract fine details from image data while maintaining a relatively high level of performance (convergency rate) as well as accuracy through its skip connections. A model with pretrained parameters for image recognition was deployed. A visual layout of the network is present below:
![Image of architectural layout of ResNet50](https://cdn.prod.website-files.com/645cec60ffb18d5ebb37da4b/65eb303c9ee4b67135628e9a_archi.jpg)
The ResNet outputs a 2048-dimensional vector.
An additional sequence of fully-connected (FC) layers followed by a classifier output is applied to the output of the ResNet. The series of FC layers is implemented for finer control in feature extraction and simplification for classification. Each layer halves the number of features, from 2048 down to 256. The classifier is a singular FC layer with features 256 -> 2, producing the binary output for class label predictions.
### 3.2.2 Loss Evaluation
Due to the nature of the network, two different loss functions were combined to optimise all aspects of the model.
1. Cross-Entropy Loss
   - Cross-Entropy Loss is used to monitor learning on the binary classification itself, penalising the model for incorrect class labelling.
2. Triplet Loss
   - Triplet loss is used to monitor the Siamese network ability to detect dissimilarities, penalising the model for incorrect pairings
   - Requires a base 'anchor', a positive image belonging to the same class as the anchor, and a negative image that doesn't
   - Evaluates similarity numerically as the Euclidean distance between images
### 3.2.3 Other Specifications
- Adam optimiser used with an initial learning rate of 1e-3, which is within the recommended lr range for this particular optimiser
- ReduceLROnPlateau lr scheduler for dynamic adjustment of learning rate over time to maximise performance
- Autocasting for mixed precision training to dynamically adjust network and output numerical parameters to optimise runtime
- Model hyperparameters (i.e. training epoch count, batch size etc.) can be modified in `config.yaml`
# 4. Results and Analysis
Run `python3 predict.py` to run the evaluation script.
## 4.1 Evaluation Metrics
Numerous performance metrics were used for the analysis of training as well as correctness of results.
1. Combined Loss
   - Combined triplet loss and classification loss over time for both training and evaluation datasets. The closer this value to 0, the lesser the loss of both the dissimilarity and classification generated by the model.
2. Accuracy
   - Numerical representation of accuracy of the model's classification outputs. Evaluated as the percentage of classifications that were 'correct', which is determined whether the model output of the correct class probability exceeds 0.5.
3. AUC-ROC
   - Area Under the Receiver Operating Characteristic Curve. Measures the model's classification outputs across all probability thresholds. More flexible than the general accuracy score. The closer this value to 1, the more accurate the model predictions.
4. Confusion Matrix
   - Detailed numerical breakdown of model performance by class, displaying amount of samples for each combination of class label and predicted label. Provides information on false positive and negative rates which is useful for specofoc analysis on what combinations of data causes the most misclassifications.
## 4.2 Results Analysis
### 4.2.1 Training and Validation Plots
![Training and Validation plots for 16 epochs](plots/metrics.png?raw=true)
Above is the training, validation, and AUC-ROC plots over 16 epochs of training and validation.
All training plots show relatively stable trends over epochs. The loss value decreases gradually, while both accuracy and AUC-ROC increases over time. Validation loss and accuracy showed similar trends, however is significantly more unstable, with noticeable spikes/dips at epoch 2 and 11. Validation AUC-ROC is much more stable, increasing over time with small fluctuations towards the end of training. Some signs of plateuing is also present in the validation AUC-ROC plot, suggesting that additional training will likely be detremental to the model and lead to overfitting.
### 4.2.2 Confusion Matrix
![Confusion matrix for validation data](plots/confusion_matrix.png?raw=true)
Above is the confusion matrix for the validation dataset. A general numerical analysis of the results is as follows:
| Category | Percentage |
| -------- | ---------- |
| True Positive (Classified=Benign, Label=Benign) | 81.96% |
| False Positive (Classified=Malignant, Label=Benign) | 18.04% |
| True Negative (Classified=Malignant, Label=Malignant) | 72.65% |
| False Negative (Classified=Benign, Label=Malignant) | 27.35% |

Within the context of this dataset, the main category of interest is True Negative, where malignant cases are misclassified as benign. This causes the main concern as it is the category with the most detrimental impact in practice, as misdiagnosed patients would be wrongly informed of their health and potentially miss important medical treatment. The value of 72.65% is considerably lower than the 80% goal, indicating the necessity of further improvements on the model.
### 4.2.3 ROC Curve
![ROC curve for validation data](plots/roc.png?raw=true)
A final AUC-ROC score of 0.85 was reached on validation data. The blue ROC curve shows the model is highly capable of identifying dissimilarities between classes, though some instabilities are still present, demonstrated by the parts of the curve that do not follow the overall contour.
# 5. References