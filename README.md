# Vector Quantised AutoEncoder trained over the HipMRI study for Prostate Cancer Radiotherapy
## 1.0 Model Description
A vector quantised autoencoder is an expanded version of the traditional encoder-decoder (variational autoencoder) architecture.
This model was developed in PyTorch by employing an object-oriented programming approach. It features an encoder for deriving a latent representation of a MRI scan, a quantisation layer for embedding the image and a decoder for reconstructing a similar image 
from the stored embedding. 
Trained over 10,000 MRI scans of the male pelvis provided as a part of the "HipMRI study for Prostate Cancer Radiotherapy", it achieves an SSIM score of 0.67 at generating images of various
scans representing occurences of a pelvis with different stages of prostrate cancer.
## 2.0 Application(s)


## 3.0 Architecture & Process
This model was first presented in 2017 through the paper "Neural Discrete Representation Learning" (Oord et al, 2018) and demonstrated a small yet
significant change which involved encoding the inputs as discrete representations rather than continuous ones. For example, when this VQVAE was trained over MRI scans of
the pelvis to reconstruct realistic scans, it encoded each image into a discrete vector stored within a dictionary of embeddings, known as a codebook. Instead of predicting
a normalised distribution of these MRI scans, the model encodes pixels into a categorical distribution of indices in vector space, which are then referenced by the decoder to
reconstruct images.

## 4.0 Bibliography
Kang. J. (2024). _Pytorch-VAE-tutorial_. https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

Malysheva, S. (2018). _Pytorch-VAE_. https://github.com/SashaMalysheva/Pytorch-VAE

Oord, A. v. d., Vinyals, O. & Kavukcuoglu, K. _Neural Discrete Representation Learning_. https://doi.org/10.48550/arXiv.1711.00937

Yadav, S. (2019, September 1). _Understanding Vector Quantized Variational Autoencoders (VQ-VAE)_ [Blog]. https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a
