# Generative Model for Brain Image Synthesis using StyleGAN

## Description
This project implements a generative adversarial network (GAN) architecture, specifically a modified version of StyleGAN, to synthesize high-quality brain images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset. The model employs various innovative layers such as Pixel Normalization, Adaptive Instance Normalization, and Weighted Sum Linear layers to effectively generate images that mimic real brain scans.

## Problem Statement
Generating realistic brain images is critical for advancing research in neuroimaging and Alzheimer's disease. This project addresses the challenge of creating high-fidelity synthetic images that can supplement existing datasets, aiding researchers in understanding disease progression and identifying biomarkers.

## How It Works
The architecture consists of a Generator and a Discriminator, operating in an adversarial manner. The Generator creates images from latent vectors, while the Discriminator evaluates their authenticity. 

## Input and Output
- **Input**: Latent vectors sampled from a normal distribution.
- **Output**: Generated brain images in the specified output directory.

## Pre-processing Steps
- **Image Normalization**: Images were resized to 256x256 pixels and normalized to [0, 1].
- **Augmentation**: Random transformations (flipping, rotation) were applied to enhance dataset diversity.

### Generator

The Generator is responsible for creating synthetic images from random noise. It utilizes a progressive growth approach, allowing it to generate images at varying resolutions. Key components include:

- **Mapping Network**: Transforms latent vectors (noise) into a style space using a series of weighted sum linear layers and pixel normalization.
  
- **Adaptive Instance Normalization (AdaIN)**: Adjusts the mean and variance of the generated images based on style vectors, allowing for fine control over the generated output.

- **Noise Injection**: Introduces random noise into the generation process to enhance diversity and realism.

- **Progressive Blocks**: A series of generator blocks that build the final image progressively, each consisting of weighted sum convolutional layers, AdaIN, and leaky ReLU activations.

- **Fade-In Technique**: Smoothly transitions between different resolutions during the image generation process to avoid artifacts.


### Discriminator

The Discriminator evaluates the authenticity of images (real vs. fake). It employs a similar progressive growth architecture, enabling it to discern finer details as resolution increases. Key components include:

- **Convolutional Blocks**: Consists of weighted sum convolutional layers and leaky ReLU activations, progressively processing the input images.

- **Minibatch Standard Deviation**: Calculates the standard deviation of feature maps across a batch, enhancing the model's ability to detect variations in generated images.

- **Fade-In Technique**: Similar to the generator, this technique is used to combine outputs from different resolutions.


## File Structure
```
/project-directory
│
├── modules.py        # Model components: Generator and Discriminator
├── dataset.py        # Data loading and preprocessing
├── train.py          # Training, validation, and saving the model
├── predict.py        # Usage example for generating images
├── embedding.py      # Script for t-SNE embedding visualization
└── README.md         # Project documentation
```

## Dependencies
- Python >= 3.7
- PyTorch >= 1.8
- NumPy >= 1.19
- torchvision >= 0.9
- Matplotlib >= 3.3
- scikit-learn >= 0.24  # For t-SNE visualization

## Training Process
- The training involves loading the dataset and progressively increasing the image size.
- The losses for both the Generator and Critic are recorded, and visualizations are generated post-training.
- The trained models are saved for future inference.

## Visualizations
### Loss Metrics
- Loss metrics for both the Generator and Discriminator are plotted during training to monitor convergence. These plots provide insights into the training dynamics and help assess model performance.

### t-SNE Embedding
- A t-SNE embedding visualization is generated to analyze the distribution of latent space representations. This helps in understanding how well the model captures the diversity of the training data.

### Generated Image Grids
- A 9x9 grid of generated images is created for both Alzheimer's Disease (AD) and Normal Control (NC) classes. This provides a visual assessment of the quality and variety of the synthesized images, showcasing the model's ability to generate realistic brain scans.

## Conclusion
This project demonstrates the capabilities of GANs in generating realistic brain images, providing a valuable tool for researchers in the field of neuroimaging. Future work may include fine-tuning the model and exploring additional image generation techniques.