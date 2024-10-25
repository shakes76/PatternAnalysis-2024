# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

# Task Statement:

Classify Alzheimer’s disease (normal and AD) of the ADNI brain data (see Appendix for link) using one of the latest vision transformers such as the GFNet [1] set having a minimum accuracy of 0.8 on the test set.

# Introduction:

Someone in the world develops dementia ever 3 seconds [2]. Alzheimer's Disease is a progressive neurodegenerative disorder that affects memory. Diagnosis of this disease consist of MRI scan to identify the potential areas that are affecting the patient. Machine learning / Deep learning models can help assist the diagnosis by being able to "learn" how to recgonise specific structures to increase diagnosis speeds. As early diagnosis is crucial for patient care.

Some traditional methods are Convolutional Nueral Networks (CNNs) for image classicaction however Vision Transformers (ViT) models like GFNet are developing better insight due to their learning pipeline specifically related to self-attention and frequency domains.


Transformer models use self-attention layers to capture long-term dependencies, which are able to learn more diverse interactions between spatial locations[1].
Global Filter Network (GFNet), which follows the trend of removing inductive biases from vision models while enjoying the log-linear complexity in computation. The basic idea behind this architecture is to learn the interactions among spatial locations in the frequency domain.

    
# Library Dependencies:

# Modules.py
- Architecture of the Visual Transformer model.
- Imports Torch and Torchvision.
- Implements multi-layer perceptron (MLP), GlobalFilter, Blocks, patchembedding
- Greyscale images
- Normalisation of Mean and SD to 0.6
- Data Augmentation 

# dataset.py
- This module handles data loading and preprocessing.
- Constructs dataloaders for training, testing and validation

# Train.py
- Houses training loop and Evaluation loop
- Has print statements for each epoch to keep track of progress

# predict.py
- This module serves as the entry point for running the project.
- Imports the train module, evaluation model for model training.
- Creates the Visual Transformer GFNet
- Calls the training function to train the model.
- Visualizes training and validation accuracy and loss.
- Sets hyperparameters.

# Description of Model

Mechanism:
    Fourier Transform: Converts spatial data into the frequency domain using 2D FFT.
    Learnable Global Filter: Applies a learnable filter in the frequency domain to modify specific frequency components.
    Inverse Fourier Transform: Converts the filtered data back to the spatial domain using inverse FFT.

Input Image (224x224)
        ↓
DataLoader & Transforms
        ↓
Batch of Images (B, 1, 224, 224)
        ↓
Patch Embedding (Conv2d)
        ↓
Embedded Patches (B, num_patches, embed_dim)
        ↓
Positional Encoding
        ↓
Embeddings (B, num_patches, embed_dim)
        ↓
[ GFNet Block × N ]
        ↓
Layer Normalization
        ↓
Global Average Pooling
        ↓
Classification Head (Linear)
        ↓
Predictions (AD or NC)

GlobalFilterLayer:
    This class converts the input from the spatial domain to the frequency domain, resulting in complex-valued frequency representations.

Block
    this class processes input embeddings through normalization, global filtering, feedforward networks, and residual connections.

GFNet
    Stacks multiple GFNetBlock instances to build a deep network capable of hierarchical feature learning.
    this class processes input images through patch embedding, multiple GFNet blocks, normalization, and a classification head to perform image classification.

Data Transforms
     Enhances model generalization and ensures consistent input dimensions and normalization. Also produces some data augmentaiton like horizontal flipping, rotaitons and nomralises the data into a mean of 0.5 and standard deviation of 0.5 

# Evaluation of Model

# Training loop 
The training loop iterates for a specified number of epochs,
Total test samples: 9000
Training set size: 18292
Validation set size: 3228
Test set size: 9000

### Validation
The model's performance is assessed during training using a validation dataset, which is separate from the training data 15% the total size of the test. This allows for monitoring how well the model generalizes to unseen data.

# Results

First test run

Epoch 30/30
----------
Train Loss: 0.0024 Acc: 0.9998
Val   Loss: 0.1377 Acc: 0.9616
Best model updated with accuracy: 0.9616

Best Validation Accuracy: 0.9616
Training completed.

=== Evaluating Model on Test Set ===
Test Accuracy: 0.6458
Final Test Accuracy: 0.6458

Testing accuracy was 6458% which is well below the training and Validation tests. This was due to the low samples for training a ViT. With in the next major run i added data augmentation techniques like horizontal Flip and rotate to increase the gernalisability in hopes the model can learn better. I also changed from 30 epochs to 150 with a batch size of 64 instead of 128 to get find better minimas. 

Epoch 150/150
----------
Train Loss: 0.0401 Acc: 0.9869
Val   Loss: 0.1066 Acc: 0.9644

Best Validation Accuracy: 0.9650
Training completed.

=== Evaluating Model on Test Set ===
Test Accuracy: 0.6667
Final Test Accuracy: 0.6667

THis is after the second run. With no significant change in my test accuracy this leads me to believe that my model is over fitting. WHich leads me wanting to decrease the dimension and depth wihtin my GFNet to not over learn the feature.

This code is able to be run on rangpur potentiall with greater epochs, more augmentation, other hyper parameter fine tuning to get desired accuracy. 


This code is able to be run on rangpur potentiall with greater epochs, more augmentation, other hyper parameter fine tuning to get desired accuracy. 


## Accuracy & Loss
- Caveat of small test 30 EPochs - i had rangpur dependency issue for 5 days 
Train Loss: 0.0024 Acc: 0.9998
Val   Loss: 0.1377 Acc: 0.9616

=== Evaluating Model on Test Set ===
Test Accuracy: 0.6458
Final Test Accuracy: 0.6458

Second Run 

Epoch 150/150
----------
Train Loss: 0.0401 Acc: 0.9869
Val   Loss: 0.1066 Acc: 0.9644

Best Validation Accuracy: 0.9650
Training completed.

=== Evaluating Model on Test Set ===
Test Accuracy: 0.6667
Final Test Accuracy: 0.6667




# References

[1] Y. Rao, W. Zhao, Z. Zhu, J. Zhou, and J. Lu, “GFNet: Global Filter Networks for Visual Recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10 960–10 973, Sep. 2023, conference Name: IEEE Transactions on Pattern Analysis and Machine Intelligence. [Online]. Available: https://ieeexplore.ieee.org/document/10091201?denied=

[2] https://www.alzint.org/about/dementia-facts-figures/dementia-statistics/

[3] https://github.com/raoyongming/GFNet