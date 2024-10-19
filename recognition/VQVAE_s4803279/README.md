# COMP3710 Project Farhaan Rashid s4803279

## Background
The purpose of this of this model is to learn from historical data and be able to create new images through the use of machine learning. The model I am using is a Vector Quantised Variational Auto Encoder (VQVAE2). This model has 3 main parts, and encoder, a latent vector embedding space and a decoder. The training data is processed through the encoder to match the dimensions of the embedding space where a codebook will map each feature to a matching vector and then the decoder will essentially build back up to the original size of the image.

# Version Requirments
Python 3.12.3
Pytorch 2.4.1
TQDM
Matplotlib
Nibael
