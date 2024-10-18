# COM3710 - StyleGAN on ADNI dataset -  Documentation
This report will be completed with documentation about the styleGAN model for ADNI brain dataset in different aspects of implementation, data training and testing, analysis of the results and further evaluation of the project.  
## The Problem
This project involves implementing **StyleGAN** on the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** brain dataset to generate realistic brain MRI images. The task is to train a GAN (Generative Adversarial Network), specifically the **StyleGAN**, which uses style modulation and progressive growing to generate high-quality images. The primary challenge is to generate meaningful and realistic MRI scans by progressively learning style-based features. The model combines **noise injection, adaptive instance normalization (AdaIN), and modulated convolutions '(Affine Transformation)'** to generate synthetic brain images.
## Dataset
The dataset used in this project is the ADNI dataset, which contains MRI brain scans categorized into different classes such as Alzheimer's Disease (AD) and Normal Control (NC). These scans are used for training the GAN to generate images that resemble real brain MRIs. The dataset is split into training and testing sets located at (Rangpur):
* Training set: /home/groups/comp3710/ADNI/AD_NC/train
* Testing set: /home/groups/comp3710/ADNI/AD_NC/test

The dataset is processed in 2D, and images are resized to 128x128 pixels before being fed into the model.
## Requirements
## Code Structure
The code is structured into the following components:
* dataset.py: Handles the loading of the ADNI dataset and applies necessary transformations like resizing and normalization.
* modules.py: Contains the main components of StyleGAN, including:
    * Mapping Network: Transforms the latent vector z into an intermediate vector w.
    * Modulated Convolution: Convolutions modulated by the style vector w.
    * AdaIN (Adaptive Instance Normalization): Adjusts feature maps using the style vector.
    * Noise Injection: Adds controlled noise at different layers.
    * Generator and Discriminator: Builds the full GAN architecture.
* train.py: Contains the training loop, where the model is trained over multiple epochs, and images are generated after each epoch.
* predict.py: Carry out the implementation of TSNE and UMAP embeddings plot with ground truth in colors
## Model Implementation
## Training/Testing & Results