# StyleGAN for ADNI (Alzheimer's Disease Neuroimaging Initiative)

#### COMP3710 - Pattern Analysis 2024
**Task 8** - Generative model of the ADNI brain dataset using StyleGAN (blend of 1 and 2).<br>
**Author:** Ethan Laskowski (46984863)

## Project Overview

This project implements a StyleGAN model for generating brain MRI images, specifically trained on the ADNI dataset. 
This projects goal is to create a Generator/Discriminator complex which yields a generative model capable of producing realistic looking brain MRI images.
This project has deep-rooted real-world connections, as the ability to generate MRI images could be used in classification (owing to the
relatively small sample set of real Alzheimer's brain scans), or in another similar applications.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Data Setup and Preprocessing](#data-preprocessing)
7. [Model Architecture](#model-architecture)
8. [Training Process](#training-process)
9. [Results](#results)
10. [Analysis of Results](#analysis)
11. [Visualisations](#visualisations)
12. [Performance Metrics](#performance-metrics)
13. [References](#references)

## Project Structure

The project consists of several key files:

- [`train.py`](train.py): Main training script for the Stable Diffusion model
- [`generate_images.py`](generate_images.py): Script for generating new images from trained model, alongside loss plots.
- [`modules.py`](modules.py): Contains Generator, Discriminator, MappingNetwork and other models.
- [`utils.py`](utils.py): Utility functions for getting noise, converting latent space to style space etc
- [`dataset.py`](dataset.py): Loads ADNI data.

## Dependencies
- Both Linux and Windows are supported.
- 64-bit Python 3.10 or later installation. Anaconda3 is recommended due to its intra-library compatibility installation.
- Additional libaries include -

    |  Libaries       |Version           |
    |-----------------|------------------|
    |  pytorch        |2.0.1             |
    |  torchvision    |0.15.2            |
    |  pytorch-cuda   |11.8              |
    |  cudatoolkit    |10.1              |
    |  numpy          |1.22.3            |
    |  matplotlib     |3.7.2             |
    |  tqdm           |4.66.1            |

    Note: Libaries may use newer version. If using conda, install pytorch, torchvision and cuda [together](https://pytorch.org).
- The training was performed on NVIDIA A100 40GB vGPU with 128GB of DRAM.

## Installation

To set up the project, follow these steps:

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Single usage as shown below (does not support cmd args parsing). Refer to [config.py](config.py) for resetting parameters. \
The network is trained from scratch, the generator and discriminator loss is printed on each iteration.
```
> python train.py

Python 3.10.13
Device:  cuda

  0%|          | 0/954 [00:00<?, ?it/s]
  0%|          | 0/954 [00:01<?, ?it/s, gp=0.992, loss_critic=10.1, loss_gen=6.17, plp=0.01]
  0%|          | 1/954 [00:01<19:10,  1.21s/it, gp=0.992, discrim_loss=10.1, gen_loss=6.17, plp=0.01]
  0%|          | 1/954 [00:01<19:10,  1.21s/it, gp=0.986, discrim_loss=9.1, gen_loss=5.98, plp=0.01] 
  0%|          | 2/954 [00:01<14:34,  1.09it/s, gp=0.986, discrim_loss=9.1, gen_loss=5.98, plp=0.01]
  0%|          | 2/954 [00:02<14:34,  1.09it/s, gp=0.978, discrim_loss=8.67, gen_loss=6.38, plp=0.01]
  0%|          | 3/954 [00:02<13:06,  1.21it/s, gp=0.978, discrim_loss=8.67, gen_loss=6.38, plp=0.01]
  0%|          | 3/954 [00:03<13:06,  1.21it/s, gp=0.97, discrim_loss=8.32, gen_loss=5.66, plp=0.01] 
  0%|          | 4/954 [00:03<12:22,  1.28it/s, gp=0.97, discrim_loss=8.32, gen_loss=5.66, plp=0.01]
  0%|          | 4/954 [00:04<12:22,  1.28it/s, gp=0.963, discrim_loss=8.03, gen_loss=6.58, plp=0.01]
  1%|          | 5/954 [00:04<11:57,  1.32it/s, gp=0.963, discrim_loss=8.03, gen_loss=6.58, plp=0.01]
  1%|          | 5/954 [00:04<11:57,  1.32it/s, gp=0.955, discrim_loss=7.63, gen_loss=4.7, plp=0.01] 
  1%|          | 6/954 [00:04<11:43,  1.35it/s, gp=0.955, discrim_loss=7.63, gen_loss=4.7, plp=0.01]
  1%|          | 6/954 [00:05<11:43,  1.35it/s, gp=0.943, discrim_loss=7.35, gen_loss=5.52, plp=0.01]
  1%|          | 7/954 [00:05<11:34,  1.36it/s, gp=0.943, discrim_loss=7.35, gen_loss=5.52, plp=0.01]
```
Sample images are generated and saved every 2 epochs and are stored as outlined in preprocessing.
Models are saved every 4 epochs and are stored as outlined in preprocessing.
Plot for the generator and discriminator loss over all iterations are saved at the end of training.

## Dataset
The Alzheimer's Disease Neuroimaging Initiative (ADNI) is a large-scale, longitudinal research study designed to develop \
clinical, imaging, genetic, and biochemical biomarkers for the early detection and progression of Alzheimer's Disease (AD). \
Launched in 2004, ADNI is funded by both public and private organizations and aims to provide a comprehensive dataset that \
supports research on Alzheimer's disease, mild cognitive impairment (MCI), and cognitive normal aging. \
The dataset specification can be found below.

|  Attributes     | Values           |
|-----------------|------------------|
|  Total images   | 30520            |
|  Image size     | 256x240          |
|  Image color    | Grayscale        |

More information is available from https://adni.loni.usc.edu/

## Data Preprocessing

The ADNI dataset should be organised in the following structure:

```
home/
    groups/
        comp3710/
            ADNI/
                AD_NC/
                   train/
                        AD/
                        NC/
                    test/
                        AD/
                        NC/ 
```
Note that the data is loaded in as a folder - being the AD_NC folder.

Where AD represents Alzheimer's Disease samples and NC represents Normal Control samples. 

Any model checkpoints will need to be placed in their appropriate folder:
```
Models/
    Gen/
    Discriminator/
    MappingNetwork
```

Any images will be saved in the following folder:
'''
saved_examples/
    256x256/
'''

### Data augmentation

Here are some image samples before augmentation.
<p align="center">
    <img src="images_for_readme/1011824_83.jpeg" alt="Image 1" width="19%" />
    <img src="images_for_readme/1011824_84.jpeg" alt="Image 2" width="19%" />
    <img src="images_for_readme/1011824_85.jpeg" alt="Image 3" width="19%" />
    <img src="images_for_readme/1011824_86.jpeg" alt="Image 4" width="19%" />
    <img src="images_for_readme/1011824_87.jpeg" alt="Image 5" width="19%" />
    <br>
    Images before Augmentation
</p>


**Resize**: The imported images were resized to 256x256 from their abnormal 240x256 shape. \
**RandomVerticalFlip**: The images were flipped vertically randomly at 50% probability to introduce variabiliity in data - preventing overfitting. \
**Normalise**: Images were normalised with mean 0.5 and standard deviation of 0.5 for each channel to convert to [0, 1] data range.  \
**Grayscale**: Since the default import for img dataset is 3 channels, when using 1 channel the images are transformed to grayscale.

Few examples of images after augmentation is shown in the figure below.

<p align="center">
    <img src="images_for_readme/999708_102.jpeg" alt="Image 1" width="19%" />
    <img src="images_for_readme/999708_103.jpeg" alt="Image 2" width="19%" />
    <img src="images_for_readme/999708_104.jpeg" alt="Image 3" width="19%" />
    <img src="images_for_readme/999708_105.jpeg" alt="Image 4" width="19%" />
    <img src="images_for_readme/999708_106.jpeg" alt="Image 5" width="19%" />
    <br>
    Images after Augmentation
</p>


## Model Architecture
StyleGAN1 (2019) is a generative adversarial network (GAN) developed by NVIDIA, designed to generate high-resolution images \
with a focus on disentangled and controllable image synthesis. A key innovation is the style-based generator, which uses a \
mapping network to transform latent vectors into style vectors, allowing for fine control over features like pose, texture, \
and shape across different levels of detail.

StyleGAN2 (2020) builds on this, addressing issues like "droplet artifacts" in the generated images by redesigning normalization \
and architectural components, including weight demodulation. It improves image quality and training stability, producing more \
realistic and diverse outputs with enhanced control over style attributes and better detail consistency.

<p align="center">
    <img src="Original-StyleGAN-architecture-a-The-latent-vector-z-introduced-b-eight-fully.png" alt="augmented images" width="35%">
    <br>
    Discriminator's architecture
</p>



### Generator Network
The generator network is divided into two sub-networks; the mapping network and the synthesis network. Both of these networks
are learnable and are optimised by the Adam optimiser in the code implementation.

**1. Mapping Network:**
Initialized using z_dim (latent space) and w_dim (style vector) as parameters, the mapping network consists of eight EqualizedLinear layer, that equalizes the learning rate, with ReLU as their activation function. Latent space dimension is initialized using pixel norm.

The mapping network converts latent space vectors (z) (which are just Gaussian noise - which is used as the basis of image generation (via the Generator) in non-style GAN)
into a new space, the style space (or W). The problem with the latent space vectors is that they are entangled, meaning a single change to z can result in a whole host
of changes in the generated image. Conversely, the style space (W) is (ideally) a disentangled feature space, where a change to a style space vector corresponds to a
particular feature change. This mapping network is a learnable network and, consequently, gets increasingly better at converting to style vectors which are
disentangled, allowing for better differentiation. These style space vectors are used as noise injections throughout the architecture, as outlined below.

**2. Synthesis Network:**
The synthesis network is where the actual image generation happens, building images step by step from low to high resolution. It is composed of several blocks that progressively refine the image resolution. The key innovations here include the use of style modulation, adaptive instance normalization (AdaIN), and noise injection.

**2.1. Progressive Structure**
Progressive Growing: The synthesis network starts with a small learned constant (of size 4x4 in this code implementation) and then progressively
increases the resolution by doubling it with each successive block. 


**2.2. Style Modulation with AdaIN**
Gaussian distribution noise is converted to normalised, modulated style vectors via equalised layers and through the use of Adaptive Instance Normalisation (AdaIN). AdaIN computes the mean and standard deviation of the non-style vectors and then uses these parameters to standardise the non-style vector. From there, the known "scaling" and "shifting" functions of the style space, learned through the Mapping Network are applied to the noise. Hence, the noise becomes a style vector, which is applied to the input to allow for finer control over features.

**2.3. Noise Injection**
At each resolution, random noise is added directly to the feature maps, introducing stochastic variations. This helps the model generate images with subtle details like hair strands or skin textures. The style vectors provide global control over image features. However, with only style vectors, the generated images are minimally diverse. Hence, noise injection enables the generation of fine, localised details - creating enhanced diversity. Each block has its own learnable scaling factor to adjust the impact of the noise.

**2.4: Style Mixing**
This involves the generation of images from multiple different latent vectors (and corresponding style vectors), in order to minimise overfitting to a particular feature (generated by one style vector).

The generator is then passed through a tanh activation function to ensure pixel values are normalised in range [-1, 1].


### Discriminator Network:
The discriminator network is used to distinguish between real and generated images. It is also a CNN, and its role is to assess the authenticity of the images produced by the generator. The discriminator is trained to minimize the distinguishability of generated images from real ones.

The discriminator architecture is as follows:

**1: Downsampling:**
Each DiscriminatorBlock is responsible for reducing the spatial resolution of the input feature maps while increasing the depth (number of channels). This makes the network more sensitive to large-scale structures.
Within these downsampling blocks, weight-equalised 2d convolution is sequentially used, alternating with LeakyReLU activation functions.
Additionally, there is a residual path to bypass this main convolutional pathway, allowing for the minimisation of gradient vanishing/exploding if they occur.

**2: Minibatch Standard Deviation:**
An invention in StyleGAN2, minibatch standard deviation generates the standard deviation across a batch of input samples, and essentially rejects a sample range where there is very minimal standard deviation. This ensures that the generator must generate diverse images, rather than just finding a niche of one particularly discriminator-fooling image, and sticking with this.
