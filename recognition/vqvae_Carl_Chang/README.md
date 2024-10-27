# VQ-VAE for Image Reconstruction
## 1. Project Overview
This project implements a Vector Quantised-Variational AutoEncoder (VQVAE) to reconstruct medical images from the HipMRI Study on Prostate Cancer. The goal is to analyse the effectiveness of the model in producing high-quality reconstructions of medical scans. This aims address the problem of efficient image compression and reconstruction, since the algorithm provides a way to encode images into a more compact, discrete representation whilst preserving essential features. The images can be reconstructed by the model using the learned quantized latent space, reducing storage requirements with reduced information loss. 

The algorithm has been trained on processed 2D slices of medical scans and measures the quality of reconstruction using metrics such as Mean Squared Error (MSE), Perplexity, and Structural Similarity Index (SSIM). The goal is to achieve a reasonably clear reconstructed image with an SSIM score above 0.6.

## 2. How It Works
### Model Architecture
The project utilises a VQVAE, which consists of:
- **Encoder**: Compresses input images into lower-dimensional latent representations. It uses three convolutional layers then a residual stack, allowing the encoder reduce input size while retaining essential features.
- **Vector Quantizer**: Converts latent representations to a finite set of embeddings (or codebook vectors). Maps the continuous latent vectors to a set of discrete emebeddings called codebook vectors. It aims to minimise the difference between the latent and codebook vectors, as well as encouraging efficient use of the codebook through maximising perplexity. These discrete representations help the model learn a more structured representation.
- **Decoder**: Reconstructs images from the quantized latent vectors. It first takes these vectors and extracts the features using a convolutional layer to transform it into an image-like feature map. Then it is passed through the residual stack, then through two transpose convolutional layers to upsample the feature map, increasing the image size and bringing it back to its original resolution.

There is also a residual stack in the encoder and decoder containing residual blocks. Each block has two convolutional layers with a skip connection that adds the input of the block to its output. This architecture enables the model to learn and retain complex features without losing information, which is particularly useful for improving image reconstruction.

During training, the model learns to minimise the difference between the original images and their reconstructions. It also aims to achieve high perplexity.
By compressing the input to discrete latent representations and reconstructing the images, the VQVAE can learn meaningful features and patterns in the data, which is useful for applications like compression and image generation.

Reference - Code modules are adapted from [2]. Hyperparameters used are also the same, but epochs changed to reduce trainig time.

### Training Process
The model is trained using the following steps:
1. **Data Loading**: The dataset is already split into training, validation, and test sets, loaded using DataLoader.
2. **Forward Pass**: Input images are passed through the encoder, vector quantizer, and decoder.
3. **Loss Calculation**: The reconstruction error (MSE) and vector quantizer loss are calculated.
4. **Optimization**: The model parameters are updated using Adam optimiser to minimise the combined loss.
5. **Evaluation**: After each epoch, the model's performance is validated using the SSIM metric.

### Pre-Processing & Data Splits
- **Pre-Processing**: The images are normalized using Z-Score normalisation before being fed into the model. This helps with stabilisation during the training process and achieves better reconstructions. They are loaded using provided sample code that has been adapted to work with PyTorch's DataLoader.
- **Data Splits**: 
  - **Training Set**: Used to learn the parameters of the model.
  - **Validation Set**: Used to tune hyperparameters and assess performance during training.
  - **Test Set**: Used to evaluate the final performance of the model.

## 3. Dependencies
These are the key dependencies installed to replicate the results:
- `torch==2.5.0+cu124`
- `torchvision==0.20.0`
- `numpy==2.0.2`
- `matplotlib==3.9.2`
- `scikit-image==0.24.0`
- `tqdm==4.66.5`
- `nibabel==5.3.2`

## 4. Results
### Training Metrics
The training process tracks the following metrics:
- **Reconstruction Error (MSE)**: Measures the difference between the original and reconstructed images using mean squared error.
- **Perplexity**: Indicates how uniformly the codebook vectors are used, i.e. how diverse the usage is, which gives an idea of how well the model is using the discrete codes.
- **SSIM**: Measures the similarity between the original and reconstructed images, with a target of 0.6 or higher.

**Example Plot of Metrics Over Epochs:**

![image](https://github.com/user-attachments/assets/1770c9ac-47c0-40ef-9579-f04588532e8f)
![image](https://github.com/user-attachments/assets/2922da3e-ed57-4161-8848-1278d263e3b9)
![image](https://github.com/user-attachments/assets/848ff545-6d34-4ef0-861c-602c453090ff)


## 5. Reproducibility
The results can be reproduced using the following steps:
1. **Set Up Environment**: Create a virtual environment and install the dependencies.
2. **Data Preparation**: Ensure dataset is organized in the following structure:
    ```
    ./data/
      ├── keras_slices_train/
      ├── keras_slices_validate/
      └── keras_slices_test/
    ```
3. **Training**: Run `train.py` to train the model. The script will load the training and validation datasets, train the VQVAE model, and save the trained weights as `vqvae_model.pth`. Key losses and metrics will also be printed during training, and plotted after training. A sample of validation original and reconstructed images will then be displayed
4. **Prediction**: Run `predict.py` to generate results on the test set. The script will load the saved model weights and perform reconstructions on the test set. It will then display original and reconstructed images with an SSIM score.
### Example Input/Output
**Input**: 2D slice of a prostate MRI scan  
**Output**: Reconstructed 2D slice of the prostate MRI scan

The following figure shows the original/input (top row) and reconstructed/output images (bottom row):

![image](https://github.com/user-attachments/assets/a77f83a1-bfbb-4c6b-b8f2-ddd30e37c618)


## 6. References
- [1] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,” arXiv:1711.00937 [cs], May 2018, arXiv: 1711.00937. [Online]. Available: http://arxiv.org/abs/1711.00937
- [2] K. Rasul, pytorch-vq-vae, (2019), Github repository. Available: https://github.com/zalandoresearch/pytorch-vq-vae/tree/master

