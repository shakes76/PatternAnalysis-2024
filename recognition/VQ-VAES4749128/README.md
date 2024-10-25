# Generative VQ-VAE model on the HipMRI Study on Prostate Cancer using the processed 2D slices
## Overview
The aim of of this problem is to create a generative model of the HipMRI Study on Prostate Cancer using the processed 2D slices using a Vector Quantized Variational Autoencoder (VQ-VAE) and pixel CNN. We will be using a VQ-VAE to quantize latent codes for training a generative PXELCNN. The outputs from the generative Pixelcnn will be decoded as a way to reconstruct new images.
## VQ-VAE
A VQ-VAE model is a generative model which combines elements from vector quantization (VQ) and Variational Autoencoders(VAE). There are three key parts to this model: encoder, decoder, codebook layer.

The encoder is a neural network that processes the input images to transform it into a latent representation, capturing the essential features of the imput and reducing its dimentionality. 

Codebook Layer uses a nearest neighbor search algorithm to quantise the continuous latent vectors produced by the encoder. Each continuous latent vector is replaced by the closest vector from a finite set of embeddings called the codebook.
The aim here is to reduce the complexity of the latent space allowing the model to learn discrete representations. This allows us to enhance both training stability and sample quality.


Input data is loaded into the encoder, a neural network, and transformed into latent representation. The latent vectors are quantised into a finite set of embeddings by using nearest neighbour search to replace continous latent presentations with the closest vector from the codebook (a finite set of vectors).The final part is the decoder which essentially attempts to decode the now quantised latent variables and reconstruct the image to its original form. 

![image](https://github.com/user-attachments/assets/9ff9b52a-d84c-4b0c-9047-ea78389c3ddd)

## Pixel CNN

Pixel CNN is another generative model used for creating images. It does this on a pixel to pixel basis through auto-regression. More specifically, it models the joint probability of all the pixels in that image and generates pixels based on pixels before the current pixel.

![image](https://github.com/user-attachments/assets/59e73fd1-6e3a-41de-a8eb-3cd1e39511ba)



PixelCNN models the distribution of latent codes. VQ-VAE encodes images into a discrete latent space represented by a finite codebook. PixelCNN captures spatial patterns within this latent space by predicting each latent code based on those previously generated pixels This enhances the model's ability to produce coherent sequences.

After the model is trained, PixelCNN can now generate new latent codes that align with the learned distribution. These codes are then decoded by the VQ-VAE decoder. The usage of PixelCNN and VQ-VAE enables us to generate new images by using intricate visual features learned from the training dataset.

## Dependencies

| Dependency | Version |
| ------------- | ------------- |
| python|3.11.2 |
| torch| 2.5.0+cu121 |
| torchvision| 0.20.0+cu121 |
| numpy| 1.26.4 |
|tqdm|4.66.5|
| scikit-image| 0.24.0 |
| Matplotlib| 3.9.2|

## Usage:

``` python train.py -mode vq_vae # mode: vq_vae , pixelcnn ```
To run predict:
``` python predict.py ```

The arguments can be customised, otherwise the default will be used when running in terminal:
```
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
```

# VqVae Training

The VQVAE model was trained with the whole training set because the model is generative. The model has 5000 updates and a batch size of 32. The loss components consists of three key types: reconstruction Loss (MSE), embedding loss and total loss. The reconstruction loss is measured as the mean squared error between the original input data and reconstructed data. The total loss is the sum of embedding loss and reconstruction loss. It is used for backpropagation.
SSIM over time for VQ-VAE is shown to satisfy the requirement of SSIM> 60, the SSIM is measured using cikit-image's structural_similarity function. 
<p align="center">
  
![image](https://github.com/user-attachments/assets/1fcb31a8-0276-4aa9-867b-70d921406ecc) 
</p>


VQ-VAE training and validation loss function. It can be seen that the loss became extremely low towards the end: 
![image](https://github.com/user-attachments/assets/7049951d-7db5-490e-a876-a4fd5d8807fb)

After 5000 updates, here are a few samples of the outputs from the VQ-VAE:

![image](https://github.com/user-attachments/assets/4b8f3313-842a-4ff9-84ed-320aab53b73e)

Here are two examples of the quantised samples:
<p align="center">
  
![image](https://github.com/user-attachments/assets/d0bfb478-1fc7-477d-932b-81c06a751d51)
  
![image](https://github.com/user-attachments/assets/6a32f6f6-e4f8-4ab6-b59d-1b563f130e18)
</p>

The pixelCNN will be trained on the quantised output. The decoded image corresponding to the first of the two quantised samples shown is provided below. I t can be seen that it is still lacking. It could be possible that 200 epochs was not enough to train the pixelCNN. More hyperparameter tuning could be needed.
<p align="center">
![e6137b5a-628a-459f-9c89-115cb1a07dcd](https://github.com/user-attachments/assets/0a5b5433-92fa-4c1d-998a-aedb697fea78)
</p>

## reference
https://www.google.com/search?sca_esv=640dfa46d8859720&rlz=1C1GCEB_enAU1024AU1024&sxsrf=ADLYWIKlgn8pA_TD7EEt8MO8HtBIdIzdlA:1729856418859&q=vqvae+diagram&udm=2&fbs=AEQNm0CrHVBV9axs7YmgJiq-TjYcvrKLYvLdNLLD2b8MCfaxte6rE3yH_shvJRqV-Iqr8JJvO9luGxMyf8tABHRE_ER5WVi_ouuYD0ZGCgonp8RpBmOUpTB-X6dVFbJc8KMdvjlHxs0_OJiYCY4-Y60oHTMiC_1a9mkGkMIYHO4XqP68ipa4P5rJaQCtA4WPne6f0aAKhdyAMTPbTsWJEdFYpNvI5RzOgw&sa=X&ved=2ahUKEwj-6O6quamJAxX4lFYBHYUHIN0QtKgLegQIDxAB&biw=2089&bih=1270&dpr=1#vhid=FKT-mO4RzQzIIM&vssid=mosaic

https://paperswithcode.com/method/pixelrnn
