# Alzheimer’s Disease Classification Using Vision Transformers (ViT)

**Student Number:** 47415056

**Name:** Swastik Lohchab

**Description** 
This project focuses on classifying Alzheimer’s Disease from MRI scans using Vision Transformers (ViT). The approach leverages ViT’s ability to capture spatial correlations across different regions of the brain, aiming to identify key areas associated with Alzheimer’s Disease. This project was conducted as part of the COMP3710 course at the University of Queensland and achieved a test accuracy of 68.20%.
---

## Table of Contents
1. [How It Works](#1-how-it-works)
2. [Network Architecture](#2-network-architecture)
3. [Dependencies](#3-dependencies)
4. [Reproducibility](#4-reproducibility)
5. [How to Run](#5-how-to-run)
6. [Data Pre-Processing and Splits](#6-data-pre-processing-and-splits)
7. [Training and Evaluation](#7-training-and-evaluation)
8. [Results and Visualizations](#8-results-and-visualizations)
9. [Future Improvements](#9-future-improvements)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)


## 1. How It Works

### Overview
The Vision Transformer (ViT) model takes an input image, divides it into patches, linearly embeds the patches, and feeds them into a series of Transformer layers to extract meaningful features for classification. In this project, we train the ViT model on a dataset of MRI images, each labeled as either ‘Normal’ or ‘Alzheimer’s Disease’.

The attention mechanism of the model underlines important brain areas that correlate with AD, hence giving insight into the model's decision-making process.

### Key Steps
1. Patch Embedding: The input image is divided into 16x16 patches, which are then flattened and linearly embedded to form a sequence.
2. Positional Encoding: Positional encodings are added to the patch embeddings to preserve spatial information.
3. Transformer Blocks: The embeddings are passed through multiple Transformer blocks, each consisting of a multi-head self-attention layer and a feed-forward neural network.
4. Classification Head: The final feature vector is used for binary classification, predicting either ‘Normal’ or ‘Alzheimer’s Disease’.


## 2. Network Architecture

### Key Components of Vision Transformer (ViT)
1. Patch Embedding Layer: Converts the input image into a sequence of flattened image patches, which can be processed by the Transformer.
2. Positional Encoding: Injects information about the spatial positions of the patches into the embeddings, as the Transformer is inherently permutation-invariant and does not have a built-in notion of order or position.
3. Transformer Blocks: The final output of each Transformer block is a refined set of patch embeddings, which are then passed to the next block in the sequence.
4. Classification Head: Transforms the output of the final Transformer block into class predictions.

### Steps of the Network Architecture
1. Input Image Processing
2. Adding Positional Encodings
3. Processing Through Transformer Blocks
4. Classification

### Benefits of Using Vision Transformers (ViT)
1. Unlike CNNs, which have a limited receptive field, ViT can capture long-range dependencies across the entire image using self-attention.
2. ViT can be easily scaled up or down by changing the number of patches, the embedding dimension, or the number of Transformer blocks.
3. Attention maps provide insights into which regions of the image are most important for the model’s decision-making process, making ViT a valuable tool for medical image analysis.

![ViT Architecture](/images/vit_architecture.jpg)


## 3. Dependencies
1. torch==2.0.1
2. torchvision==0.15.2
3. numpy==1.25.0
4. matplotlib==3.7.1
5. Pillow==9.4.0
6. scikit-learn==1.2.2 for performance metrics
7. scipy==1.11.1
8. python==3.12.4


## 4. Reproducibility

### Environment
1. Hardware: Training was conducted on Rangpur High-Performance Computing with NVidia A100 GPUs.
2. Software: Python 3.12 environment, using Anaconda.


## 5. How to Run

1. Clone the repository:

```
git clone https://github.com/yttrium400/PatternAnalysis-2024.git
cd PatternAnalysis-2024/recognition/vit_47415056
```

2. Train the model:

```
python train.py
```

This will train the ViT model on the ADNI dataset and save the trained model to `model_weights.pth` within the base vit_47415056 folder.

3. Prediction:

```
python predict.py
```


## 6. Data Pre-Processing and Splits

### Pre-Processing Steps
1. 	Resizing: All MRI images are resized to 224x224 pixels to ensure uniform input size, which is compatible with the Vision Transformer model.
2. Normalization: The pixel values are normalized using a mean of 0.1415 and a standard deviation of 0.2420, helping to standardize the input data and improve model convergence.
3. Data Augmentation: Various augmentation techniques are applied to increase data variability and reduce overfitting:
	•	Random Horizontal Flip: Flips images horizontally with a 50% probability.
	•	Random Vertical Flip: Adds further variability by flipping images vertically.
	•	Random Resized Crop: Randomly crops and resizes images to add randomness to the input images.
	•	Adjusting Sharpness: Modifies the sharpness of images to simulate different imaging conditions.
4. ToTensor Conversion: Images are converted to PyTorch tensors, which is necessary for inputting data into the model.
5. Shuffling: The training data is shuffled to ensure that the model does not learn any unintended patterns based on the order of the images.

### Data Structure

```
AD_NC
├── test
│   ├── AD
│   └── NC
└── train
    ├── AD
    └── NC
```

### Image 1 -> AD
![AD image from train set](images/AD.jpeg)
### Image 2 -> NC
![NC image from train set](images/NC.jpeg)

### Splitting Strategy
The training set was further divided into 90% for training and 10% for validation. The split ensured that images from the same patient were not present in both subsets to maintain data integrity.


## 7. Training and Evaluation

### Configuration
1. Model: Vision Transformer (ViT) - "vit_small_patch16_224"
2. Optimizer: Adam with learning rate of 1e-5 and StepLR scheduler
3. Batch Size: 32
4. Number of Epochs: 11
5. Loss Function: Cross-Entropy Loss
6. Early Stopping: Triggered if validation loss did not improve for 7 epochs

### Training Loop
The training loop monitored both accuracy and loss metrics. Early stopping was implemented to prevent overfitting.


## 8. Results and Visualizations

### Performance 
The final model achieved an accuracy of 68.20% on the test set.

![Training and Testing Accuracy](images/result.jpg)

### Training and Validation Plots
1. Accuracy vs. Epochs:  ![Training and Testing Accuracy VS No. of epochs graph](images/accuracy_graph.jpg)
2. Loss vs. Epochs:  ![Training and Testing loss VS No. of epochs graph](images/loss_graph.jpg)

### Confusion Matrix
The confusion matrix provides insights into the model’s classification performance:

![Covariance Matrix](images/covariance_matrix.jpg)


## 9. Future Improvements
1. Data Augmentation: Explore additional augmentation techniques to further improve model generalization.
2. Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and Transformer configurations.
3. Attention Analysis: Conduct a more in-depth analysis of the attention maps to understand the model’s focus areas better.


## 10. Conclusion
This project successfully implemented a Vision Transformer (ViT) to classify Alzheimer’s Disease from MRI scans, achieving a test accuracy of 68.20%. While the model shows promise, the accuracy is limited by factors such as the complexity of MRI data, limited dataset size, and potential overfitting.

### Analysis of Training and Validation Graphs
1. Accuracy vs. Epochs: The accuracy graph shows a steady improvement in both training and validation accuracy over the epochs, indicating that the model is learning meaningful features from the data. However, there are noticeable fluctuations in the validation accuracy, which could be due to overfitting, where the model starts to memorize the training data instead of generalizing well to unseen data.
2. Loss vs. Epochs: The loss graph demonstrates a consistent decrease in training loss, but the validation loss does not drop as steadily. The gap between training and validation loss suggests that the model might be overfitting. Despite early stopping, there is still a risk that the model’s performance could degrade on new data, highlighting the need for more robust regularization techniques or a larger dataset.

### Analysis of the Confusion Matrix
The confusion matrix reveals additional insights into the model’s performance:

1. True Positives (Top-Left: 3790): The number of correctly classified Normal cases is relatively high, indicating that the model is good at identifying non-Alzheimer’s patients.
2. True Negatives (Bottom-Right: 2160): The number of correctly identified Alzheimer’s cases is also significant but lower than the true positives, suggesting the model is somewhat less effective at detecting Alzheimer’s.
3. False Positives (Top-Right: 670): These are cases where the model incorrectly classified Normal patients as having Alzheimer’s. While relatively fewer in number, false positives could lead to unnecessary concern or medical procedures.
4. False Negatives (Bottom-Left: 2380): The false negatives are concerning, as these represent Alzheimer’s cases that the model failed to identify. In a medical context, such errors are particularly critical, as undiagnosed Alzheimer’s Disease could delay necessary treatment or intervention.


## 11. References
1. Visual Transformer Architecture (ViT) - (https://huggingface.co/docs/transformers/model_doc/vit)
2. ADNI Dataset - (https://adni.loni.usc.edu)
3. ViT Overview - (https://www.geeksforgeeks.org/vision-transformer-vit-architecture/)
4. chatgpt - (https://chatgpt.com/c/6722567c-918c-800d-aae9-045e4d1dbf33)
