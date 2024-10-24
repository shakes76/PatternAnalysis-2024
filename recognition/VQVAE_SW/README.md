# VQ-VAE for Hip MRI Image Reconstruction
This project implements a **Vector Quantized Variational Autoencoder (VQ-VAE)** for reconstructing Hip MRI images. The model leverages advanced deep learning techniques such as residual blocks, self-attention mechanisms, and perceptual loss to improve the quality of reconstructed images. This README provides a detailed description of the algorithm, usage instructions, dependencies, and how to run the model. Additionally, it includes examples of input and output images along with result visualizations.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm Overview](#algorithm-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- Usage Instructions
  - [1. Setting Up the Environment](#1-setting-up-the-environment)
  - [2. Preparing the Dataset](#2-preparing-the-dataset)
  - [3. Training the Model](#3-training-the-model)
  - [4. Testing the Model](#4-testing-the-model)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Magnetic resonance imaging (MRI) is a medical imaging technique used in radiology to form pictures of the anatomy and the physiological processes inside the body(Wikipedia Contributors, 2019). High-quality reconstruction of MRI images is crucial for accurate diagnosis and treatment planning. This project focuses on reconstructing Hip MRI images using a VQ-VAE, which combines the strengths of variational autoencoders and vector quantization to capture complex image representations in a discrete latent space.

## Algorithm Overview

### Vector Quantized Variational Autoencoder (VQ-VAE)

The VQ-VAE is an autoencoder that uses a discrete latent space to represent data. Unlike traditional autoencoders, which use continuous latent variables, VQ-VAE maps inputs to a finite set of embedding vectors (codebook). This allows the model to learn discrete representations, which are more suitable for tasks like image reconstruction and generation.

#### Key Components:

- **Encoder**: Transforms the input image into a latent representation.
- **Vector Quantizer**: Discretizes the latent representation by mapping it to the nearest embedding vector in the codebook.
- **Decoder**: Reconstructs the image from the quantized latent representation.

### Model Enhancements

- **Residual Blocks**: Used in both the encoder and decoder to facilitate training deeper networks by adding skip connections, which help in addressing the vanishing gradient problem.
- **Self-Attention Layers**: Allow the model to focus on different parts of the input, capturing long-range dependencies and improving the representation of global features.
- **Perceptual Loss with VGG16**: Incorporates a perceptual loss calculated using a pre-trained VGG16 network to measure the difference in high-level feature representations between the original and reconstructed images.

## Dataset

The dataset consists of Hip MRI slices stored in NIfTI format (`.nii` or `.nii.gz`). It is organized into training, validation, and test subsets:

- `keras_slices_train`
- `keras_slices_validate`
- `keras_slices_test`

**Note**: You need to download the dataset and place it in the appropriate directory as specified in the usage instructions.

## Dependencies

The project requires the following packages:

- Python 3.7+
- NumPy
- PyTorch
- torchvision
- nibabel
- scikit-image (for SSIM calculation)
- Matplotlib

You can install the required packages using:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```arduino
numpy
torch
torchvision
nibabel
scikit-image
matplotlib
```

## Project Structure

- `dataset.py`: Handles data loading and preprocessing.
- `modules.py`: Defines the model architecture, including the encoder, decoder, residual blocks, self-attention layers, and vector quantizer.
- `train.py`: Script to train the VQ-VAE model.
- `predict.py`: Script to evaluate the trained model on the test set and visualize results.
- `vqvae_hipmri.pth`: Saved model weights after training (generated after running `train.py`).

## Usage Instructions

### 1. Setting Up the Environment

1. **Clone the repository:**

   ```bash
   https://github.com/SylviaWwww/PatternAnalysis-2024.git
   cd recognition
   cd VQVAE_SW
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### 2. Preparing the Dataset

1. **Download the Dataset:**

   - Download the `HipMRI_study_keras_slices_data.zip` file containing the MRI slices.

2. **Unzip the Dataset:**

   - Place the zip file in the `data/` directory within the project.
   - The `dataset.py` script will automatically unzip and organize the data when you run the training script.

   **Directory Structure After Unzipping:**

   ```kotlin
   data/
   └── HipMRI_study_keras_slices_data/
       ├── keras_slices_train/
       ├── keras_slices_validate/
       └── keras_slices_test/
   ```

### 3. Training the Model

Run the `train.py` script to start training the model:

```bash
python train.py
```

**Training Parameters:**

- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Number of Epochs**: 50
- **Perceptual Loss Weight**: 0.1

**Training Output:**

The script will print the loss after each epoch. The trained model weights will be saved as `vqvae_hipmri.pth`.

**Example Output:**

```less
Epoch [1/50], Loss: 87.1079
Epoch [2/50], Loss: 87.0585
Epoch [3/50], Loss: 86.9046
...
Epoch [50/50], Loss: 85.1023
```

### 4. Testing the Model

Run the `predict.py` script to evaluate the model on the test set and visualize the results:

```bash
python predict.py
```

**Output:**

- The script will display the original and reconstructed images for the first five samples in the test set.
- It will also compute and print the average Structural Similarity Index Measure (SSIM) across the test set.

**Example Output:**

```yaml
Average SSIM: 0.4776
```

**Visualization:**

For each of the first five test images, the script displays:

- **Original Image**
- **Reconstructed Image**
- **SSIM Score**

## Results

### Sample Reconstructions

Below are examples of the original and reconstructed images from the test set:

**Example 1:**

- **Original Image**

  <img src="/Users/siyaowu/Library/Application Support/typora-user-images/Screenshot 2024-10-24 at 11.26.17 PM.png" alt="Screenshot 2024-10-24 at 11.26.17 PM" style="zoom:50%;" />

- **Reconstructed Image**

  <img src="/Users/siyaowu/Library/Application Support/typora-user-images/Screenshot 2024-10-24 at 11.26.44 PM.png" alt="Screenshot 2024-10-24 at 11.26.44 PM" style="zoom:50%;" />

  **SSIM: 0.5074**

**Example 2:**

- **Original Image**

  <img src="/Users/siyaowu/Library/Application Support/typora-user-images/Screenshot 2024-10-24 at 11.27.52 PM.png" alt="Screenshot 2024-10-24 at 11.27.52 PM" style="zoom:50%;" />

- **Reconstructed Image**

  <img src="/Users/siyaowu/Library/Application Support/typora-user-images/Screenshot 2024-10-24 at 11.28.05 PM.png" alt="Screenshot 2024-10-24 at 11.28.05 PM" style="zoom:50%;" />

  **SSIM: 0.5198**

**Example 3:**

- **Original Image**

  <img src="/Users/siyaowu/Library/Application Support/typora-user-images/Screenshot 2024-10-24 at 11.28.22 PM.png" alt="Screenshot 2024-10-24 at 11.28.22 PM" style="zoom:50%;" />

- **Reconstructed Image**

  <img src="/Users/siyaowu/Library/Application Support/typora-user-images/Screenshot 2024-10-24 at 11.28.48 PM.png" alt="Screenshot 2024-10-24 at 11.28.48 PM" style="zoom:50%;" />

  **SSIM: 0.5261**

### Quantitative Evaluation

- **Highest SSIM Across Test Set**: `0.5261`

The SSIM scores indicate that while the model captures some structural similarities between the original and reconstructed images, there is room for improvement. The moderate SSIM values suggest that the model may benefit from further tuning or architectural enhancements to better preserve important structural details in the MRI images.

## Conclusion

This project demonstrates the implementation of a VQ-VAE with residual blocks, self-attention layers, and perceptual loss for MRI image reconstruction. While the current results show moderate success in reconstructing the images, the relatively low SSIM scores highlight the need for potential improvements. Future work could involve experimenting with different hyperparameters, increasing the size of the codebook, or incorporating additional loss functions to enhance the reconstruction quality.

## References

- **VQ-VAE Paper**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **VGG16 Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Wikipedia Contributors**, “Magnetic Resonance Imaging,” *Wikipedia*, Mar. 15, 2019. https://en.wikipedia.org/wiki/Magnetic_resonance_imaging
