# Alzheimer's Disease Classification using ViT
## The problem and how I've approached it

Alzheimer’s Disease can be diagnosed using a combination of clinical and imaging data. The algorithm designed aims to use this imaging data from the ADNI dataset to classify Alzheimer’s Disease (AD) and Normal Control (NC). 

For this, a Vision Transformer (ViT) model was used to identify patterns in brain scans that distinguish between AD and NC. 

It will train a model that can classsify AD and NC brain images. An image path can be supplied for the code to predict the classification of. It will return whether or not the predicted correctly.


## How it Works
how it works in a paragraph and a figure/visualisation.

The algorithm uses a Vision Transformer (ViT), which processes brain images by splitting them into patches. These patches are passed through transformer layers to capture relationships between the image regions. The model is trained on labeled brain scans to classify the images as either Alzheimer's Disease or Normal Control.

A simple representation of the model structure is as such:

1. Patch embedding
    - input image is divided into square patches. 
      Each patch is then linearly transformed into a vector.
      This results in a sequence of patch embeddings, which serve as the input tokens for the subsequent layers.

2. Positional embedding
    - Adds positional information
    - learned and added to the patch embeddings at the input stage
    
3. Encoder layers
    - Multi-Head Self-Attention
        - self-attention mechanism to capture the relationships between different patches in the input sequence
        - computes a weighted sum of all patch embeddings, where the weights are determined by the relevance of 
          each patch to the current one
        - keeps focus on important patches

    - Feedforward Neural Networks
        -  introduce non-linearity and allow the model to learn complex relationships between patches

    - Layer Normalization and Residual Connections
        - helps stabilize and speed up training by normalizing the inputs to each sub-layer

This model is trained in a training loop, where a loss graph is generated. Then further statistics and graphs are generated from the saved model in predict.py.

### Inputs: 
Preprocessed brain scans from the ADNI dataset (240x256 pixels), sorted into a test or train folder which each contain two folders: AD and NC.

An image file_path to compare its classification with the prediction classification

Model: Vision Transformer 
Inlcudes multiple layers, multi-head attention, and linear projection.

### Outputs
- The predicted label (AD or NC) for a given brain scan.
- Figure: Loss Curve
- confusion matrix 
- test accuracy
- sk-learn's classification report for further accuracy information

### Pre-processing
Grayscale Conversion: The brain scans were converted to grayscale (single-channel) since the images are all white/black/grey

Resizing: All images are resized to 240x256 pixels in dataset.py. This is largely unnecessay though as all images should be the correct size.

Normalization: The images were normalized using a mean and standard deviation of 0.5, which helps stabilize training.

### Data Splits:
Around 22000 training images and around 8800 test images were used. 
Training images were split such that 80% were used in training and 20% were used in validation. The test images (which were 28% of the total training set before splitting) were used for testing.

### Justification for Data Splits:
Given the plentiful dataset and the fact that we are using a deep learning model (not shallow), focusing more on training to make sure subtle medical details are properly recongised was very important. Large amounts of data are required to learn the complex patterns, and insufficient training data could cause overfitting.

The 20% split for validation would be sufficient for a reliable estimate, due to the large dataset we have access to. In smaller datasets, this would not be the case, as a 20% sample may not be enough to represent the diversity of the set.

Similarly, the test dataset, being as large as 28% of the training set, is sufficient due to the large datasets being used.


### Dependencies:

Python: 3.8 or later 

PyTorch: 1.9.0 or later

torchvision: 0.10.0 or later

scikit-learn: 0.24.2 

matplotlib: 3.4.3 

### Reproducability 
To reproduce my results of 60% accuracy, it would be necessary to download AD_NC data into the correct filepaths provided in graphs folder that has been committed to this repository. 

Upon setting filepaths for the model to save in modules.py, predict.py and train.py, and setting data_dir in dataset.py and predict.py
(and also setting the filepath for a chosen image within the data_dir)
run predict.py. It will train the model, and print a loss graph, confusion matrix and accuracy stats as well as a prediction for the given image.
Upon running train from within predict.py once, it is possible to comment out the training and just work with the saved model.


### References
TheDeepHub. (2022, March 10). Building Vision Transformer from Scratch using PyTorch: An Image worth 16x16 words. Medium. https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27

Schmid, P. (2022, April 6). Image Classification with Huggingface Transformers and Keras. https://www.philschmid.de/image-classification-huggingface-transformers-keras

Pulfer, B. (2022, June 1). Vision Transformers from Scratch (PyTorch) – A Step-by-Step Guide. Medium. https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

Pulfer, B. (2022, June 1). Vision Transformers from Scratch (PyTorch) – A Step-by-Step Guide. Medium. https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
