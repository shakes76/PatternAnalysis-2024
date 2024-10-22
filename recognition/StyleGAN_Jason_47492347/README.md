# StyleGAN Brain MRI Generative Model
This is an image generation model trained on the ADNI brain dataset with the goal of producing "reasonably clear" deepfake images of brain MRI's.


## Overview
The field of medical imaging is currently undergoing significant changes due to recent advancements in machine learning, which is being utilised for tasks such as disease detection and image enhancement. Despite this progress, there are many challenges that remain, one of which is that medical datasets are often lacking in size and diversity. Many tasks in machine learning require subtantial amounts of data to achieve meaningful solutions. This project aims to address this problem by training a generative model capable of producing images that imitate real world data, specifically brain MRI scans.


## StyleGAN Lore

### What is a GAN?
Generative Adversarial Network (GAN) is a machine learning framework developed by scientist Ian Goodfellow and his team in 2014. At a high level, it can be conceptualised as an evolutionary competition between two models, the Generator and the Discriminator. The Generator's task is to gradually improve at producing fake imitations of the training data, while the Discriminator, which is trained alongside, is tasked with evaluating whether the images it receives are from the realdata set or produced by the Generator. Over many iterations of this process, the Generator eventually becomes capable of creating realistic deepfake images (in this case) that are difficult for the Discriminator to distinguish from the original dataset.

### StyleGAN and its improvements
While traditional GANs were able to display impressive potential, a key limitation was that there was a lack of control over the outputs, especially in terms of semantics and details. The StyleGAN architecture, first introduced in a 2018 paper by Nvidia researchers, was a revision of the GAN with noticeable improvements- almost a complete restructure of the Generator's design resulted in a model that allows for much better fine-tuned control over image outputs. This was achieved by the incorporation of the concept of "style", which allows for the Generator to separate different components of the image and control them independently of one another. Instead of starting with a single latent vector $Z$, StyleGAN contains a mapping network that first transforms $Z$ into an intermediate latent space $W$, which is then used to feed adjustment inputs at different layers of the model. Through this process, we can address the problem of "entanglement" in traditional GANs, where adjusting for a certain feature of an image would affect many other unrelated features. Overall, the StyleGAN archetecture allows for more intuitive and structured control over the Generator outputs, ultimately leading to better results.

## ADNI Dataset and Data Preprocessing
The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset is a "comprehensive and widely used collection of longitudinal clinical, imaging, genetic, and other biomarker data". The specific dataset used for this project was a collection of real 2D brain MRI scans presented as side-view jpeg images. The data is divided into two categories- Normal Control (NC) for healthy brains, and Alzheimer's Disease (AD). While each subset were also divided into train/test splits initially, I combined them together into just the training dataset- this specific generative task did not require a train/test/validate split. This resulted in roughly 15,000 images per train dataset.

Simple augmentation was performed on the data- a resize to 128x128 pixels for quicker training times, a greyscale conversion to use 1 channel, and normalisation using mean=0.5, stdev=0.5 for easier computations. I decided against using traditional methods such as random rotations or flips to maintain the consistence in orientation of the images in the original dataset.

## StyleGAN Architecture Details

### Progressive Growing
The original StyleGAN paper is actually an improvement based on a preceding improvement called ProGAN, which is an archetectural change to the traditional GAN where instead of training the model at full resolution, we start at a very small image size (8x8 in this case) and progressively grow the image throughout the training process. This method has shown to be a much more stable training process.

### Mapping Network
The mapping network is a simple multi-layer perceptron (MLP). It is responsible for the aforementioned mapping of latent vector $Z$ into the intermediate latent space $W$. Initially, pixel normalization is performed on the input (following the structure given by the ProGAN paper), and it then passes through eight weight scaled linear layers separated by ReLU activation functions.

### Generator
The Generator starts takes a small random noise vector (8x8 pixels in this case) as input, and gradually converts them into higher resolution image outputs. At every image resolution, the structure contains a series of convolutional layers for feature extraction (much like a traditional GAN), Adaptive Instance Normalization (AdaIN), and random noise injections for extra variability. The AdaIN step is where the style injection happens using the $W$ space. Unlike other normalization techniques commonly used in ML such as GroupNorm/BatchNorm, Instance Normalization is done on individual "instances"- per channel for each individual image, in this case. The "Adaptive" part of this process is referring to how the mean and variance are "aligned" with the style feature maps given by the mapping network- this is how "style transfer" happens. Finally, the upsampling process from a lower to higher resolution uses a parameter we call $\alpha$, which is linearly interpolated from 0 to 1, we smoothly "fade in" new layers into the image until it has grown to the desired size. These main processes are looped until we have reached the final resolution size and all transformations have been completed.

### Discriminator
The structure of the Discriminator is left mostly unchanged from traditional GAN. One difference is that it also trains alongside the Discriminator using progressively growing images. It contains convolutional layers paired with LeakyReLU activation functions for feature extraction, gradually downsampling until it can map to a prediction vector output.

## Results


## Dependencies

## Generating Images with a Pre-Trained Model
Run the following script to generate images using a model that has already been trained and saved (you will have to train one first, and store it in the saved_models directory). Specify the model, output name, number of images to generate, and the seed.

```
python predict.py --model <model_name> --output <output_name> --n <number_of_images> [--seed <random_seed>]
```

### Example Usage:

```
python generate_images.py --model my_model --output example_outputs --n 10 --seed 69
```

The generated images will be saved to your SRC directory under the following structure:
```
predict_outputs/
|-- example_outputs/
    |-- img_0.png
    |-- img_1.png
    |-- img_2.png
    |-- ...
    |-- img_9.png
```

## References
https://arxiv.org/pdf/1710.10196

http://arxiv.org/pdf/1812.04948

https://arxiv.org/abs/1912.04958

https://adni.loni.usc.edu/data-samples/adni-data/

https://blog.paperspace.com/implementation-of-progan-from-scratch/

https://blog.paperspace.com/implementation-stylegan-from-scratch/#models-implementation