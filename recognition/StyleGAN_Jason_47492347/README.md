# StyleGAN Brain MRI Generative Model
This is an image generation model trained on the ADNI brain dataset with the goal of producing "reasonably clear" deepfake images of brain MRI's.


## Overview
The field of medical imaging is currently undergoing significant changes due to recent advancements in machine learning, which is being utilised for tasks such as disease detection and image enhancement. Despite this progress, there are many challenges that remain, one of which is that medical datasets are often lacking in size and diversity. Many tasks in machine learning require subtantial amounts of data to achieve meaningful solutions. This project aims to address this problem by training a generative model capable of producing images that imitate real world data, specifically brain MRI scans.


## The StyleGAN Architecture

### What is a GAN?
Generative Adversarial Network (GAN) is a machine learning framework developed by scientist Ian Goodfellow and his team in 2014. At a high level, it can be conceptualised as an evolutionary competition between two models, the Generator and the Discriminator. The Generator's task is to gradually improve at producing fake imitations of the training data, while the Discriminator, which is trained alongside, is tasked with evaluating whether the images it receives are from the realdata set or produced by the Generator. Over many iterations of this process, the Generator eventually becomes capable of creating realistic deepfake images (in this case) that are difficult for the Discriminator to distinguish from the original dataset.

### StyleGAN and its improvements



References:

http://arxiv.org/pdf/1812.04948

https://arxiv.org/abs/1912.04958
