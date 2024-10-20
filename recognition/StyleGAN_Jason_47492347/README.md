# StyleGAN Brain MRI Generative Model
This is an image generation model trained on the ADNI brain dataset with the goal of producing "reasonably clear" deepfake images of brain MRI's.


## Overview
The field of medical imaging is currently undergoing significant changes due to recent advancements in machine learning, which is being utilised for tasks such as disease detection and image enhancement. Despite this progress, there are many challenges that remain, one of which is that medical datasets are often lacking in size and diversity. Many tasks in machine learning require subtantial amounts of data to achieve meaningful solutions. This project aims to address this problem by training a generative model capable of producing images that imitate real world data, specifically brain MRI scans.


## StyleGAN Lore

### What is a GAN?
Generative Adversarial Network (GAN) is a machine learning framework developed by scientist Ian Goodfellow and his team in 2014. At a high level, it can be conceptualised as an evolutionary competition between two models, the Generator and the Discriminator. The Generator's task is to gradually improve at producing fake imitations of the training data, while the Discriminator, which is trained alongside, is tasked with evaluating whether the images it receives are from the realdata set or produced by the Generator. Over many iterations of this process, the Generator eventually becomes capable of creating realistic deepfake images (in this case) that are difficult for the Discriminator to distinguish from the original dataset.

### StyleGAN and its improvements
While traditional GANs were able to display impressive potential, a key limitation was that there was a lack of control over the outputs, especially in terms of semantics and details. The StyleGAN architecture, first introduced in a 2018 paper by Nvidia researchers, was a revision of the GAN with noticeable improvements- almost a complete restructure of the Generator's design resulted in a model that allows for much better fine-tuned control over image outputs. This was achieved by the incorporation of the concept of "style", which allows for the Generator to separate different components of the image and control them independently of one another. Instead of starting with a single latent vector $Z$, StyleGAN contains a mapping network that first transforms $Z$ into an intermediate latent space $W$, which is then used to feed adjustment inputs at different layers of the model. Through this process, we can address the problem of "entanglement" in traditional GANs, where adjusting for a certain feature of an image would affect many other unrelated features. Overall, the StyleGAN archetecture allows for more intuitive and structured control over the Generator outputs, ultimately leading to better results.

## StyleGAN Architecture Details

## Results

## Example Usage

## References

http://arxiv.org/pdf/1812.04948

https://arxiv.org/abs/1912.04958
