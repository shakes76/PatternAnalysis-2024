# COMP3710 StyleGAN2 MRI Image Generator
A StyleGAN2 model trained on the OASIS dataset.

## Objective
The objective is to create a generative model of one of the ADNI brain data set using a StyleGAN2
that has a reasonaly clear image. Furthermore, a UMAP embeddings plot with ground truth in colours 
is included.

## StyleGAN2 - Algorithm Description
### GAN
The goal of the generative model is to learn the underlying distribution of the training dataset. 
The generative adversarial network (GAN) performs a min-max game between two convolutional networks; the 
generator and the discriminator. The generator takes random noise input and outputs a fake image with a 
fake distribution, while the discriminator performs a binary classification task of distinguishing between 
fake and real images. The goal of the generator is to generate a fake image that highly resembles the real 
image.

### StyleGAN
StyleGAN is a generative adversarial network (GAN) architecture that uses a style-based generator for image synthesis.
In contrast to traditional GANs, StyleGAN does not start directly from a random noise vector, but uses a mapping network to
map the noise to an intermediate latent space first. Random vectors from this space and by an affine transformation become style vectors, which represent the dientangled features from the training distribution. Using adaptive instance normalisation (AdaIN),
the style vectors are directly embedded into the intermediate layers of the generator. Stochastic variation control adds 
much finer details into the image through the addition of noise input into the generator similarly using AdaIN blocks.
The StyleGAN allows for style mixing, since the styles are separated across different layers which are progressively trained 
using different data sources. This allowing for better control over high-level and low-level features independently, making it possible to blend, interpolate, or adjust specific aspects of the generated image. 

### StyleGAN2
StyleGAN2 builds upon the original StyleGAN architecture by addressing several limitations, such as visible blob-like artifacts on the images and the progressive growing problem. StyleGAN2 removes the artifacts by replacing the AdaIN layer with weight modulation. Instead of manipulating the feature maps using AdaIN, the convolution kernel weights are scaled
with a style vector in weight modulation and then the kernel is normalised in weight demodulation. This removes the progressive growing problem which introduces the blob-like artifacts. The resulting architecture produces higher quality images than the StyleGAN.


## The Problem
stuff




