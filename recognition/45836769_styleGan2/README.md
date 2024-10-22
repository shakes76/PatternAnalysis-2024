# StyleGAN2 for ADNI Dataset: Alzheimer's and Normal Brain Image Generation

## Problem Statement

This project implements a StyleGAN2 architecture to generate 256x256 pixel grayscale brain images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to create a generator capable of producing realistic brain images for both Alzheimer's Disease (AD) and Normal Control (NC) categories.

StyleGAN2 on the ADNI dataset can contribute to important problems:
1. It can generate synthetic brain images that can be used for data augmentation in medical imaging studies.
2. Its a tool view the latent space of brain images, to see transition phases and structural differences between AD and NC brains.

## StyleGAN2 Architecture and Algorithm

The first StyleGAN introduced the concept of disentangled latent spaces for better control over image features like style and structure. This was done by separating a mapping network which takes a latent vector z from a Gaussian distribution and transforms it into an intermediate latent space w. This makes it easier to control individual attributes (e.g. colour, texture, or shape) without changing others. From w the synthesis network increases the resolution while applying style via Adaptive Instance Normalisation (AdaIN). AdaIN normalises the convolution layer activations using the mean and variance of the feature maps and then scales and shifts them based on learned parameters from the latent vector w.

StyleGAN had some issues with atrifacts in its images. StyleGAN2 extends on StyleGAN by replacing the AdaIN with modulated convolution. Where w is scaled by a learned transform which is then used to scale the kernel weights - this changes layer behavour based on the style vector similarly to AdaIN.


1. **Mapping Network**: Transforms the input latent code and class labels into an intermediate latent space (w).

2. **Synthesis Network**: Generates images. Each block uses modulated convolutions to apply style and upsample.

3. **Discriminator**: Evaluates realism of generated images. It uses downsampling blocks and includes a minibatch standard deviation layer and residual connections.

4. **Progressive Growing**: The network starts generating at a low resolution (8x8) and progressively grows (5 layers) to the final resolution (256x256).

5. **Modulated Convolution**: Used in the synthesis network to apply style information at each layer. Controls the generated images across all resolutions.

6. **Residual Network**: Used in the discriminator to downsample images before classification. Blocks are made up of two convolution layers (with activations) added to the input through a skip connection.

The training process involves alternating between generator and discriminator updates. The generator aims to produce increasingly realistic brain images, while the discriminator classifies real and generated images. This aims to remain balanced between the two models so both continue to learn until the generated images are satisfactory.

Below are graphs of the generator and discriminator architectures:

### Generator Architecture

```mermaid
graph TB
subgraph "Synthesis Block"
    direction TB
    MC1["Modulated Conv2D"] --> NI1["Noise Injection"]
    NI1 --> A1["Activation"]
    A1 --> MC2["Modulated Conv2D"]
    MC2 --> NI2["Noise Injection"]
    NI2 --> A2["Activation"]
end

subgraph "Generator"
    direction TB
    Z["Input Latent Z"] --> MN["Mapping Network"]
    L["Class Label"] --> MN
    MN --> W["Intermediate Latent W"]
    W --> SB1["Synthesis Block 1"]
    W --> SB2["Synthesis Block 2"]
    W --> SB3["Synthesis Block 3"]
    W --> SB4["Synthesis Block 4"]
    W --> SB5["Synthesis Block 5"]
    SB1 --> SB2
    SB2 --> SB3
    SB3 --> SB4
    SB4 --> SB5
    SB5 --> FC["Final Conv"]
    FC --> Out["Output Image 256x256"]
end
```

### Discriminator Architecture

```mermaid
graph TB
subgraph "Residual Block"
    direction TB
    In1["Input"] --> C1["Conv2D"]
    In1 --> SC["Skip Connection"]
    C1 --> A1["Activation"]
    A1 --> DS["Downsample"]
    SC --> DS
    DS --> Add["Add"]
end

subgraph "Discriminator"
    direction TB
    In["Input Image 256x256"] --> IC["Initial Conv"]
    IC --> RB1["Residual Block 1"]
    RB1 --> RB2["Residual Block 2"]
    RB2 --> RB3["Residual Block 3"]
    RB3 --> RB4["Residual Block 4"]
    RB4 --> RB5["Residual Block 5"]
    RB5 --> MSD["MiniBatch StdDev"]
    MSD --> FC["Final Conv"]
    FC --> FL["Flatten"]
    FL --> LL["Linear Layer"]
    LL --> Out["Output Score"]
end
```


## Requirements
| Package     | Version  |
|-------------|----------|
| matplotlib  | 3.9.1    |
| torch       | 2.2.2    |
| torchaudio  | 2.2.2    |
| torchvision | 0.17.2   |
| umap-learn  | 0.5.6    |
| pillow      | 11.0.0   |

## Results

