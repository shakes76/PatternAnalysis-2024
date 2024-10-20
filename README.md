# Vector Quantised AutoEncoder trained over the HipMRI study for Prostate Cancer Radiotherapy
## 1.0 Model Description
A vector quantised variational autoencoder (VQVAE) is an expanded version of the traditional variational autoencoder (VAE) architecture.
The following model was developed in PyTorch by employing an object-oriented programming approach. It features an encoder for deriving a latent representation of a MRI scan, a quantisation layer for embedding the image and a decoder for reconstructing a similar image 
from the stored embeddings. 
Trained over 10,000 MRI scans of the male pelvis provided as a part of the "HipMRI study for Prostate Cancer Radiotherapy", it achieves an SSIM score of 0.67 at generating images of various
scans representing different stages of prostrate cancer.
## 2.0 Reasons for Implementing a VQVAE over a VAE
This model was first presented in 2017 through the paper "Neural Discrete Representation Learning" (Oord et al, 2018) and demonstrated a small yet
significant change which involved encoding the inputs as discrete representations rather than continuous ones. For example, when this VQVAE was trained over MRI scans of
the pelvis to reconstruct realistic scans, it encoded each image into a discrete vector stored within a dictionary of embeddings, known as a codebook. Instead of predicting
a normalised distribution of these MRI scans, the model encodes pixels into a categorical distribution of indices in vector space, which are then referenced by the decoder to
reconstruct images. This added layer is helpful for avoiding a problem known as 'posterior collapse' in variational autoencoders (Yadav, 2019), which occurs when a continuous representation of data in lower dimensional space (i.e. after encoding the input image) proliferates a significant amount of noise - leading to the decoder generating images closer to the mean of the distributed data. If the MNIST dataset containing grayscale handwritten digits was being used to trained, a varitional autoencoder would be appropriate as this represents a continuous set of data. Even if these MRI scans are 2D and grayscale in nature, they represent different stages and spread of prostrate cancer, which represents discrete data, and therefore a VQVAE is more powerful at distinguishion when generating such images.
## 3.0 Architecture & Process
![vqvae_architecture drawio](https://github.com/user-attachments/assets/55baf8e7-8fd7-4c85-ac17-030914947c14) 
                        _Figure 1: VQVAE architecture._

Similar to the VAE, an input image (in this case a MRI scan) is provided to the encoder, which maps it to a lower latent representation, known as a feature map. After this step, instead of being passed straight to the decoder, the feature map passes through a quantisation layer, where it is compared against a previously stored dictionary of embeddings. The distance between each encoded vector and embedding is calculated, and the encoded vector is replaced with its nearest embedding. This is referred to as quantisation, and this quantised output is passed to the decoder. The decoder then maps this quantised representation to a high dimensional space and backpropagates the gradients straight to the encoder. 

It is important to note that alongside calculating a reconstruction loss value, the VQVAE calculates an additional three loss values:
- **Codebook Loss**:
A variation of the _k_-means clustering algorithm is employed here, where a group of vector embeddings in the dictionary are treated as clusters and the embedded vector representation is adjusted accordingly. Codebook loss refers to the amount of change in the interpolation distance between an embedded vector and the closest mean of cluster of embeddings, undertaken by the VQVAE during quantisation.
- **Commitment Loss**:
Similar to codebook loss, however instead of moving the embedded vectors closer to the nearest cluster, the clustered embeddings are adjusted according to the embedded vector space provided by the encoder. Otherwise, the dictionary may grow in size and fail to allocate any embeddings to the feature map. This represents a tradeoff within the VQVAE, as both codebook and commitment loss values must be treated equally by the model (Yadav, 2019). 
- **Perplexity**: 
Refers to the utilisation rate of the dictionary. A low value represents that the embedding space was referred at a low rate by the model, which is typical during the first few epochs when training. Ideally, a higher value is achieved as the model becomes better at generating images.

## 4.0 Instructions for Deployment and Dependecies Required

## 5.0 Training Plots

## 6.0 Actual vs Reconstructed Scans (can also be found in the vqvae folder).


## 7.0 Bibliography
Kang. J. (2024). _Pytorch-VAE-tutorial_. https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

Malysheva, S. (2018). _Pytorch-VAE_. https://github.com/SashaMalysheva/Pytorch-VAE

Oord, A. v. d., Vinyals, O. & Kavukcuoglu, K. _Neural Discrete Representation Learning_. https://doi.org/10.48550/arXiv.1711.00937

Yadav, S. (2019, September 1). _Understanding Vector Quantized Variational Autoencoders (VQ-VAE)_ [Blog]. https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a
