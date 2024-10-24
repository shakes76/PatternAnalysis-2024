# StyleGAN2 Based Image Generation Network on ANDI Brain Dataset
## Project Aim
The aim of the model is to create a Image generation network of the ANDI dataset based on StyleGAN2.

## Dependencies

    |  Libaries       |Version            |
    |-----------------|-------------------|
    |  python         |3.11.9             |
    |  torch          |2.5.0              |
    |  torchvision    |0.20.0             |
    |  pytorch-cuda   |11.8               |
    |  cudatoolkit    |10.1               |
    |  numpy          |1.23.5             |
    |  matplotlib     |3.9.2              |
    |  tqdm           |4.66.5             |
    |  umap-learn     |0.5.4              |
    |  pyqt           |5.15.10            |



## StyleGAN2 architecture

This project implements StyleGAN2. StyleGAN2 is an improvement from StyleGAN, which is based off of Generative Adversarial Networks (GANs). A GAN is composed 
of two main components: a generator and discriminator. Generator will generate fake images given an input noise, and the
discriminator is a classifier that classifies whether the given input image is real or fake.

StyleGAN improves GAN by using the following:
- Instead of a traditional latent input, generator takes a learned constant. 
- Using a standalone mapping network and AdaIN for style transfer.
  - The mapping network is a neural network composed of 8 fully connected layers. It takes a randomly sampled point from
  the latent space and generates a style vector. The style vector is then transformed and incorporated to each block of
  generator via adaptive instance normalization (AdaIN). 
- Addition of noise to each layer.
  - The noise is used to generate style-level variation.
- Mixing regularization
  - Since the Style generation uses intermediate vector at each level of synthesis, the network may learn the 
  correlation between the levels. Mixing regularization is introduced to reduce the correlation between levels.
![stylegan.png](model_images%2Fstylegan.png)

StyleGAN2 improves StyleGAN by addressing the shortcoming of StyleGAN, namely, the droplet effect and the preference of 
features over a certain position. Weight modulation and demodulation is used, and got rid of progressive growing
![stylegan2.png](stylegan2.png)
## Algorithm
The architecture have 3 main components: Discriminator, Generator, and Noise Mapping Network.
These three components are defined in module.py as classes. 

The Generator is made up of multiple generator blocks and a 
StyleBlock. The GeneratorBlock is made up of two connected StyleBlock and followed by a ToRGB block.

The Discriminator is a simple classification neural network with Adam optimizer.


Adam was used as the optimizer for generator, discriminator and the mapping network.
Hyperparameter used include batch_size = 32, learning rate = 1e-3, LAMBDA_GP (for Adam optimizer) = 10. The model 
generates images with resolution 256x256. This is chosen as it is closest to the original image resolution, which is 
240x256.

## Preprocessing
The images are cropped to 256x256, and then random horizontal flip was applied.

## Usage
Before training or generating examples using trained model, please download the ADNC dataset. Replace all instances of 
{AD_NC} below with the directory of your downloaded dataset.

To train on the entire dataset (without separating AD and NC classes, run:

```python3 train.py --dataset_dir {directory} --classes all```

To train specifically only on AD or NC brain images, run:

```python3 train.py --dataset_dir {directory} --classes AD``` or

```python3 train.py --dataset_dir {directory} --classes NC```

Models are saved to model folder under current directory by default. If you wish to save model and load model from a
different directory, add your directory with 

```--model_dir {directory}```

example usage:

`python3 train.py --model_dir model --classes AD --dataset_dir AD_NC/train`

To see examples using a trained model, run 

```python3 predict.py --dataset_dir AD_NC/train --model_dir model --plot_umap True --AD_dir AD --NC_dir NC```

If plot_umap is set to be True, --AD_dir and --NC_dir must be specified.

## Training result
### Comparison of fake images and real images
![epoch_0.png](fake_vs_real%2Fepoch_0.png)
![epoch_20.png](fake_vs_real%2Fepoch_20.png)
![epoch_30.png](fake_vs_real%2Fepoch_30.png)

### training loss
In total, 3 models were trained. A general model that trained on both AD and NC, an AD-only model, and a NC-only model.

#### Loss of general model
![training_loss.png](training%2Ftraining_loss.png)
![training_loss_proportion.png](training%2Ftraining_loss_proportion.png)

#### Loss of AD model
![training_loss_AD.png](training%2Ftraining_loss_AD.png)
![training_loss_proportion_AD.png](training%2Ftraining_loss_proportion_AD.png)

#### Loss of NC model
![training_loss_NC.png](training%2Ftraining_loss_NC.png)
![training_loss_proportion_NC.png](training%2Ftraining_loss_proportion_NC.png)

### UMAP embedding
Aside from a model that trained on both AD and NC brain images indiscriminately, two additional models were trained on 
only AD brain images and only NC brain images. The style embedding output by the mapping network was plotted using umap.

When the model was initialized, the umap embedding formed 2 different clusters, this is expected as the mapping network 
started with random values.

As the training progresses, the embedding starts to get closer and overlap with each other. However, as training 
progresses(starting at around epoch 30), the embedding starts to separate from each other and eventually formed 2 
separate clusters again.

This indicates that brain with AD and NC brain have different "styles". However the two distinct cluster contradicts the
recent UQ work (Liu, S. et al., 2023) where the embedding of the two disease have overlaps. 

This may indicate that the model is over-fitting, as that AD and NC model were only trained on the train set of each 
class, and there was no validation of the model using a validation step. Further investigation is needed.

![umap_ep_0.png](umap%20plot%2Fumap_ep_0.png)
![umap_ep_5.png](umap%20plot%2Fumap_ep_5.png)
![umap_ep_30.png](umap%20plot%2Fumap_ep_30.png)
![umap_ep_80.png](umap%20plot%2Fumap_ep_80.png)

## Reference 
Ian J. Goodfellow et al. (2014). Generative Adversarial Networks https://arxiv.org/pdf/1406.2661 

Tero Karras et al. (2018). A Style-Based Generator Architecture for Generative Adversarial Networks https://doi.org/10.48550/arXiv.1812.04948

Jason Brownlee (2020). A Gentle Introduction to StyleGAN the Style Generative Adversarial Network https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/

Abd Elilah TAUIL (2023). Understanding StyleGAN2 https://blog.paperspace.com/understanding-stylegan2/

Liu, S. et al. (2023). Style-Based Manifold for Weakly-Supervised Disease Characteristic Discovery. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14224. Springer, Cham. https://doi.org/10.1007/978-3-031-43904-9_36