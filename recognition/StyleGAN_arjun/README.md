# Generating Synthetic Brain Scans with StyleGAN (COMP3710)

This project implements StyleGAN (Generative Adversarial Network) to create high-quality, synthetic
brain scan images. Developed as part of the COMP3710 course, this implementation aims to demonstrate
the potential of deep learning in medical imaging synthesis.

## Objective

Our project harnesses StyleGAN technology to generate high-quality, synthetic brain scan images, addressing several critical
challenges in biomedical research and healthcare AI applications. The medical imaging field often struggles with data scarcity,
which limits the development and validation of AI algorithms. By generating synthetic brain scans, we can augment existing
datasets, providing researchers with a broader range of imaging data for their work.

These synthetic images serve multiple purposes beyond just expanding datasets. They can be instrumental in training and testing
AI models designed for various neurological applications, including disease detection, progression tracking, and treatment
response prediction. Additionally, our approach helps address the underrepresentation of rare neurological conditions in existing
datasets by generating synthetic examples of these uncommon cases. The generated images also serve as valuable educational
resources for medical students and professionals, offering diverse brain imaging examples without requiring actual patient scans.

## GAN Architecture

Generative Adversarial Networks (GANs) represent a sophisticated class of deep learning models designed for generative tasks. At
their core, GANs consist of two primary networks working in opposition: a Generator that creates synthetic data from random noise,
and a Discriminator that works to distinguish between real and generated images. These components engage in a complex minimax
game during training, where the Generator continuously improves its ability to create realistic images while the Discriminator
becomes increasingly adept at detecting synthetic ones. This adversarial process results in progressively higher quality
synthetic images.

### Feature Entanglement in GANs

One of the challenges in traditional GAN architectures is feature entanglement. This refers to the
phenomenon where different features or attributes of the generated images are not clearly separated in
the latent space. Feature entanglement can manifest in several ways:

- **Lack of Control**: When features are entangled, it becomes difficult to manipulate specific attributes
of the generated images independently. Changing one aspect of the image often affects others
unintentionally.

- **Limited Diversity**: Entanglement can limit the diversity of generated samples, as the model struggles
to combine features in novel ways.

Addressing feature entanglement is crucial for improving the quality and utility of generated images,
especially in sensitive applications like medical imaging. This is where advanced architectures like
StyleGAN and StyleGAN2 come into play.

## StyleGAN

StyleGAN, introduced by NVIDIA in 2020, represents a significant advancement in GAN architecture, designed to generate
high-quality, controllable images. The architecture introduces several innovative components that work together to improve both
the quality and controllability of generated images.

<p align="center">
<img src="./assets/StyleGAN_arch.png" width=50% height=50%
class="center">
</p>

Key components and innovations of the StyleGAN architecture include:

1. **Mapping Network**: This network transforms the input latent code z into an intermediate latent code w.
The mapping network allows for better disentanglement of features and more control over the generated
images' styles. It consists of multiple fully connected layers.

2. **Synthesis Network**: This is the main part of the generator that actually produces the image.
It's structured as a series of convolutional layers, each operating at a different resolution.

3. **Adaptive Instance Normalization (AdaIN)**: This technique applies the style at different resolutions,
enabling fine-grained control over image features. AdaIN layers are inserted after each convolutional
layer in the synthesis network.

4. **Noise Injection**: Random noise is added at each layer of the synthesis network, introducing stochastic
variation and improving the realism of generated images, especially in terms of fine details.

5. **Progressive Growing**: StyleGAN starts training on low-resolution images and progressively increases the
resolution during training, allowing for stable training of high-resolution images.

6. **Style Mixing**: During training, styles from two latent codes are randomly mixed, improving the network's
ability to separate different aspects of the generated images.

The StyleGAN family has undergone several iterations, with StyleGAN2 being a significant improvement over
the original.

## StyleGAN2

In this project, we implemented the StyleGAN2 architecture, which addresses several shortcomings of
the original StyleGAN. Key features of StyleGAN2 include:

<p align="center">
<img src="./assets/StyleGAN2_arch.png" width=50% height=50%
class="center">
</p>

1. **Modulation-demodulation**: At its core, StyleGAN2 replaces the AdaIN operations with a new modulation-demodulation mechanism.
This approach effectively eliminates the characteristic "blob" artifacts that were sometimes visible in images generated by the
original StyleGAN, resulting in smoother, more natural-looking outputs.

2. **Weight Demodulation**: Weight demodulation serves as another crucial innovation, replacing the instance normalization used
in the original architecture. This technique normalizes feature maps more effectively, leading to better overall image quality
and more stable training. The result is more consistent image generation with improved fine detail preservation, which is
particularly important for medical imaging applications where accuracy is paramount.

3. **Path Length Regularization**: This new regularization technique ensures smoother transitions in the latent space, leading to
more consistent image generation and better interpolation between different styles.

These improvements collectively make StyleGAN2 a more robust and reliable architecture for generating synthetic brain scan images,
providing better control over the generation process while maintaining high image fidelity and anatomical accuracy.
