
# Pattern Recognition and Analysis
## Classification of Alzheimer's Disease (Normal or AD) using Transformers GFNet6

## Problem Description
- Alzheimer's Disease is a neurodegenerative disorder that affects millions worldwide. For this, diagnosis and intervention should be as early as possible to ensure effective care. The project primarily related to the classification of AD vs. NC brain MRI images through the use of a **GFNet6 architecture**-based machine learning model. This model has adopted a Transformer-based architecture for vision tasks; the architecture allows for high accuracy in the classification of medical imaging.
- The dataset used in the development of this project is the **ADNI** dataset, which stands for Alzheimer's Disease Neuroimaging Initiative. This is a very standard dataset under medical research. Our goal is to achieve at least 80% or higher classification accuracy on the test dataset.

## Dataset Description
The ADNI (Alzheimer's disease Neuroimaging Initiative) dataset is widely used dataset in Medical research and Machine Learning, especially for the studies related to the Alzheimer's disease. It was created to support the discovery for the biomakers for the early detection and progression of Alzheimer's disease. The dataset includes various types of data collected from patients, ranging from imaging data to genetic information and clinical assessments. 

## Key Features of the ADNI Dataset : 
### 1) Imaging Data
- Magnetic Resonance Imaging (MRI) : In general, high-resolution structural imaging of the brain is conducted to analyze brain anatomy.
- Positron Emission Tomography(PET) : A type of functional imaging to evaluate activity of the brain; this usually targets amyloid and tau protein accumulation in the brain-perhaps the most well-known hallmarks of Alzheimer's disease. 

### 2) Clinical and Demographic Data : 
- Participants' age, sex, education, and medical history are given, which may be useful in trying to correlate the clinical outcomes with Alzheimer's disease.

### 3) Diagnostic Lagels : 
- For the dataset given to us had two categories which are mentioned below.
- Cognitive Normal (CN) : People who are healthy and in whom there is no evidence of cognitive impairment.
- Alzheimer's Disease (AD) : This group consists of patients with a diagnosis of Alzheimer's disease.

### 4) Longitudinal Data :
- The dataset also includes longitudinal follow-ups where patients data is collected over time. This will help in the essential for studying the disease progression and effectiveness of interventions.

## Algorithm Description
Traditional CNNs are very effective at extracting the spatial features of images via convolution and pooling operations. For this project, our approach follows the architecture of a GFNet-a traditional CNN combined with a Global Filter Layer that captures the local and global features of MRI images. The Global Filter Layer in GFNet adds the capability for capturing long-range dependencies across feature maps, which is particularly important for recognizing subtle patterns that can span larger areas of the brain. Fully connected layers after feature extraction carry out the classification into Cognitive Normal(CN) and Alzheimer's Disease(AD).

## Working of Algorithm & Architecture
Each convolutional layer in the GFNet model is followed by a max-pooling layer for downsampling and a dropout layer to regularize the feature maps, consequently avoiding overfitting. The gray-scale MRI image of size 128×128×1 pixels serves as input to the model. In this image, convolutions extract local spatial features such as texture and edges. It includes several convolutional and pooling operations, after which it applies the Global Filter Layer to further refine feature maps capturing global patterns of an entire image. These refined features go through a fully connected layer for classification. Non-linearity within this model is introduced with the use of the Exponential Linear Unit activation function, enhancing convergence during training. The model also applies dropout in order to reduce overfitting for randomly dropped units during training. The last output layer will predict which class the image belongs to-NC or AD.

## GFNet Architecture
The following image shows the architecture of the GFNet model used for Alzheimer's Disease classification.
![GFNet Architecture](https://drive.google.com/uc?export=view&id=1_MtDGSkJuh3BRswejXlqUUzrtikPBpcN)

## Pre-Processing Steps
- **Grayscale Conversion** :  The MRI images are converted into grayscale images. This reduces computational complexity.
- **Resizing** : Images were resized to 128x128 pixels for the model input.
- **Normalization** : It normalizes the pixel values to fall in a range with a mean of 0.5 and a standard deviation of 0.5.
- **Data Augmentation** : This includes augmenting the training data through various random horizontal flips, rotations, and random size crops. The object is to get better generalization for the model.

## Data Splitting Justification
- **Training Split(70%)** : This set provides training to the model to learn the pattern in the data.
- **Validation Split(15%)** : This set is used for tuning the hyperparameters to avoid overfitting.
- **Testing Split(15%)** :  This is a separated, unseen dataset used in order to determine the performance of a model.
The splitting of data would thereby enable the model to be validated during training, while completely new data are tested with this model, making it robust and generalizable.

## Training & Validation Loss, Training & Validation Accuracy
These two plots represent the training and validation loss (on the left) and training and validation accuracy (on the right) over the 200 epochs of training the GFNet model on the ADNI dataset.

The following image shows the training and validation loss and accuracy over 200 epochs.
![Training and Validation Loss & Accuracy](https://drive.google.com/uc?export=view&id=1FmPLCZAj04GeWjKof4ZDv9u1wwmTRhBH)

## Explanations or the Interpretations of the plots 
### Left Plot : Training & Validation Loss
- Generally speaking, both the training and validation losses are going down, which means that the model is learning well. However, the oscillations of the validation loss beyond roughly 100 epochs may suggest that the model has begun to memorize the data points from the training dataset and may be overfitting.

### Right Plot : Training & Validation Accuracy
- Both training and validation accuracies have already increased heavily in the first couple of epochs to a high accuracy of roughly 90% and do not increase further as the training proceeds.
- The closeness of training and validation accuracies is an excellent omen; this suggests that the model generalizes rather well on this validation set, and not highly overfitting, despite the small jitters in the validation loss.

### Conclusion
- It appears to be going well with an approximate accuracy of 90% on both the training and validation sets.
- A proper learning would be with a loss decrease with increased accuracy, but the fluctuations in validation loss might reflect overfitting and require further tuning for generalization, which may involve early stopping or regularization.

## Classification Report and Confusion Matrix
The following classification report and confusion matrix illustrate the performance of the model on the test dataset. The model achieved an overall accuracy of **79%** with a test loss of **0.9818**.

### Classification Report
- **CN(Cognitive Normal)** : Precision = 0.87, Recall = 0.68, F1-Score = 0.76
- **AD(Alzheimer's Disease)** : Precision = 0.74, Recall = 0.90, F1-Score = 0.81

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NC    | 0.87      | 0.68   | 0.76     | 4460    |
| AD    | 0.74      | 0.90   | 0.81     | 4550    |
| **Accuracy** |  |        | 0.79     | 9010    |

### Confusion Matrix
The confusion matrix shows the number of correct and incorrect classifications for each class.
![Confusion Matrix & Classification Report](https://drive.google.com/uc?export=view&id=1BQiFRU75TPWbFZArv8u9WKXDfmNE8DNU)

- **3018** Cognitive Normal(CN) were rightly classified and **1442** were misclassified as AD.
- **4092** Alzheimer's Disease(AD) patients were classified correctly, while **458** were misclassified as Cognitive Normal(CN).

## Requirements
- **Python**: 3.8+
- **TensorFlow or PyTorch**: Depending on your implementation choice (GFNet6 model in PyTorch)
- **NumPy**: 1.19+
- **Matplotlib**: 3.3+
- **Scikit-learn**: 0.24+
- **Google Colab Pro+**: Recommended for faster training and inference.

## Environment & Hardware Setup
- The project is developed and trained in **Google Colab Pro+** to make use of enhanced computational resources with high-performance GPUs. This has allowed for not only faster training but also faster experimentation with the model, enabling to iterate efficiently to achieve high accuracy in a much shorter period of time.
- It was trained for **200 epochs**, using a GPU instance from **Colab Pro+**. This drastically reduced the time taken to run the model when compared to local machines or basic Colab environments. This high-performance environment was important to train deep learning models like the GFNet, which require extensive computational power while processing large image datasets like those of the ADNI dataset.

## References
- Giuliano Giacaglia. (Mar 11, 2019). How Transformers Work. Towards Data Science. https://towardsdatascience.com/transformers-141e32e69591
- Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global filter networks for image classification [ArXiv preprint arXiv:2107.00645].  
https://arxiv.org/abs/2107.0064
- OpenAI. (2024). ChatGPT [AI language model]. Available from https://www.openai.com



