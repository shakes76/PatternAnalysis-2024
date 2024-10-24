# COMP3710 StyleGAN for ADNI Brain Image Generation

An implementation of StyleGAN that is used to generate brain images from the [ADNI](https://adni.loni.usc.edu/about/) dataset, while also trying to classify whether the brain images fall under Cognitive Normal (CN) or Alzheimer's Disease (AD).

Created by: **Matthew Lockett [46988133]**
***

## Table of Contents

[StyleGAN](#stylegan)

## Code Structure

Here is a summary of each item contained within this repository.

* `Assets\`: Used to store all figures and images required for the README.md.
* `dataset.py`: Used to load the ADNI dataset.
* `hyperparameters.py`: Contains all hyperparameters and configuration constants used to train and visualise the StyleGAN model.
* `modules.py`: Contains the source code implementation of the StyleGAN model and other training functions.
* `README.md`: This document, which contains all necessary information to understand this repository.
* `predict.py`: Loads the saved StyleGAN models and  from them creates a visualisation of the UMAP.
* `train.py`: Implements the main training loop for the StyleGAN and saves them.
* `utils.py`: Contains helper functions used in various places.

## The Problem

The Alzheimer's Disease Neuroimaging Initiative ([ADNI [1]](#references--acknowledgements)), is a scientifically funded study to track the progression of Alzheimer's disease using various biological markers. One of these biological markers is simply the observed changes in the brain, detected through the use of MRI or PET scans. Now as the development of deep neural networks have increasingly become more effective and reliable in the last decade or so, they have reached a point where they can now start to detect these biological markers themselves, without the need of human intervention. Therefore, the goal of this project was to utilise the StyleGAN deep neural network architecture, to both distinguish between Cognitive Normal (CN) and Alzheimer's Disease (AD) ridden brains, and also to produce generated images of them.

## The GAN Architecture

To first understand the StyleGAN architecture, it is imperative that the predecessor to it's creation, the Generative Adversarial Network ([GAN [2]](#references--acknowledgements)), is briefly described, see [Figure 1](#figure-1---the-gan-architecture-3) for a depiction of it's architecture. The GAN consists of two convolutional deep neural networks, the Generator and the Discriminator, that are actively at odds with each other (hence Adversarial). The Generator's objective is to take an input latent space vector, $z$, filled with random noise, and through multiple convolutions and fully connected layers, convert it into an image that resembles the dataset it was trained on. It's opponent, the Discriminator, is purely focused on image classification, wherein it has to decide if the image it has received is either real, from the dataset, or fake, created by the Generator. Therefore, to effectively train these two networks together they must both be of equal strength, or else the min-max game that they play will be one-sided.

#### *Figure 1 - The GAN Architecture [[3](#references--acknowledgements)]*

![GAN Architecture](Assets/GAN%20-%20GEN%20and%20DISC.png)

## The StyleGAN Architecture

The Style Generative Adversarial Network, or [StyleGAN [4]](https://arxiv.org/abs/1812.04948), builds upon the GAN model, specifically the Generator, with an emphasis on being able to dictate the style of it's image generation. This model first came about in 2018, when researches at NVIDIA realised the important role the latent vector space, $z$, played in image generation, and learnt how to filter it's noise to control various features of an image [[5](#references--acknowledgements)]. It should also be mentioned that the baseline model for the StyleGAN is actually the Progressive Growing GAN ([ProGAN [6]](#references--acknowledgements)), which instead of training the GAN on a single image size, like $64 \times 64$, the ProGAN architecture implements training on smaller resolutions first, like $4 \times 4$, and incrementally increases the size until the full resolution is trained on. This forms the base of the StyleGAN as it provides greater stability with training to larger image sizes, when compared to the original GAN. However, the underlying architecture of ProGAN is very much the GAN, and so the GAN still holds great influence over the StyleGAN.

### The Generator

The breakthrough component of StyleGAN is certainly it's improvement over the original Generator architecture as used in GAN, see [Figure 2](#figure-2---the-stylegan-generator-architecture-6). StyleGAN's Generator splits the traditional Generator into two sequential networks, the Mapping Network and the Synthesis Network. The Mapping Network takes a randomised latent space vector, $z$, and passes it through a series of eight fully connected layers to create a new intermediate space, called the style vector $w$ [[5](#references--acknowledgements)]. In the original paper for StyleGAN, both $z$ and $w$ had dimensions of 512, with each of the dimensions in $w$ meant to represent a tunable feature of an image [[5](#references--acknowledgements)]. Essentially, the eight fully connected layers of the Mapping Network are utlised to extract the underlying features hidden within the noise of the latent space vector $z$.

The Synthesis Network does not have a direct input, it instead utilises a constant $4\times4\times512$ learnable feature map to start it's underlying process. The feature map is then combined with a noise injection, B, which based on the original paper, is utilised throughout all blocks of the Synthesis Network to improve the finer details of the output image. After the noise injection is applied, the style vector from the Mapping Network is utilised and combined with an Adaptive Instance Normalisation (AdaIN) layer, with which the output of the feature map is standardised to a Gaussian distribution, and the style vector is incorporated as a scaling and shifting term. Lastly, the resultant feature map is passed through a simple $3\times 3$ convolution. These series of actions all contribute to one layer of the Synthesis Network, with two layers making an entire block. Within the StyleGAN paper, they utilised a total of nine blocks to go from the constant feature map size of $4\times 4$ to a $1024 \times 1024$ image. And through the use of the ProGAN architecture, not all blocks were used immediately, blocks were instead added with increasing image sizes.

#### *Figure 2 - The StyleGAN Generator Architecture [[7](#references--acknowledgements)]*

![StyleGAN Architecture](Assets/StyleGAN%20Architecture.png)



### The Discriminator

StyleGAN made no changes to the original ProGAN implementation of the Discriminator [[4](#references--acknowledgements)], which can essentially be explained through the use of the GAN Discriminator, as seen in [Figure 1](#figure-1---the-gan-architecture-3). The StyleGAN Discriminator is comprised of a single network and takes as an input an image either from the dataset or created by the Generator. It then applies a series of convolutions to that image to form a feature map, and then downscales the feature map within each layer. The final output of the Discriminator is thus a probability on if the given input image is real or fake. Each block within the Discriminator corresponds to a given input feature map size, and thus the ProGAN implementation adds increasingly more blocks, for each increase in image size.

## This Implementation of StyleGAN

## Results

### Example Inputs

## Dataset Structure

```
ADNI_DATASET/
    |--- Training Set/ 
    |          | --- AD/
    |          |      |--- img1.png
    |          |      |--- img2.png
    |          |      |--- img3.png
    |          | --- CN/
    |                 |--- img1.png
    |                 |--- img2.png
    |                 |--- img3.png
    |--- Validate Set/ 
    |          | --- AD/
    |          |      |--- img1.png
    |          |      |--- img2.png
    |          |      |--- img3.png
    |          | --- CN/
    |                 |--- img1.png
    |                 |--- img2.png
    |                 |--- img3.png
```

## How to Run

To run this project, you will need the following dependencies:

- Python 3.9
- PyTorch 1.10.0
- CUDA 11.1
- Matplotlib 3.4.2
- NumPy 1.21.0
- Tensorboard 2.7.0

Make sure to install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Future Improvements


## References & Acknowledgements

[1] - ADNI: <https://adni.loni.usc.edu/about/> </br>
[2] - Original GAN Paper: <https://arxiv.org/abs/1406.2661> </br>
[3] - GAN Architecture Image: <https://www.researchgate.net/figure/Network-architecture-generator-top-discriminator-bottom-The-GAN-is-composed-by_fig1_335341342> </br>
[4] - StyleGAN Paper: <https://arxiv.org/abs/1812.04948> </br>
[5] - StyleGAN Overview: <https://medium.com/@arijzouaoui/stylegan-explained-3297b4bb813a> </br>
[6] - ProGAN Paper: <https://arxiv.org/abs/1710.10196> </br>
[7] - StyleGAN Generator Image: <https://www.researchgate.net/figure/Comparison-between-a-traditional-GAN-StyleGAN-and-StyleGAN2-generator-15-16_fig5_352096439> </br>
