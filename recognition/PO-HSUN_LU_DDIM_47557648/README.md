# COMP3710 - Pattern Recognition Report

**Student Number:** 47557648

**Chosen project:** Task 8 - Create a generative model of one of the ADNI brain data set (Diffusion Model)

## Back Gournd of the Diffusion Model

AI image generation is a technology that has been hotly discussed in the art and Deep Learning (DL) field. You must have heard of the AI Art Generator such as Dall-E 2 or NovelAI, a DL model that generates realistic-looking images from a given text sequence. To explore this technology deeper, we need to introduce a new class in the generative model called ‘diffusion’, first proposed by Sohl-Dickstein et al. (2015), which aimed to generate images from noise using a backward denoising process.

So far, several generative models exist, including GAN, VAE and Flow-based models. Most of them could generate a high-quality image, such as StyleGAN-1, 2, the State-of-the-Art image generation model before diffusion model appears. However, each has some limitations of its own.

 > GAN models are known for potentially unstable training and less diversity in generation due to their adversarial training nature. VAE relies on a surrogate loss. Flow models have to use specialized architectures to construct reversible transforms (Lilian Weng, 2021)

The diffusion model has provided a slow and iterative process when noise is converted into an image; this makes the diffusion model more scalable than the GAN model. Besides, since the target of the diffusion model is to predict the input noise, which is supervised learning, we could expect the training of the diffusion model will be much more stable than GAN (unsupervised learning).

## What is Diffusion

 <p align="center">
 <img width="500px" src="https://github.com/Yukino1010/Diffusion_Model/blob/master/image_source/diffusion.png" >
 </p>
 
Diffusion refers to the movement of substances from a region of higher concentration to a region of lower concentration. Inspired by this concept, diffusion models define a Markov chain that gradually adds random noise to an image. This Markov chain can be viewed as a diffusion process, where the act of adding noise represents the "movement." Our goal is to determine the noise (movement) added to the image and reverse this process. 

The diffusion model mainly consists of two phases: Forward Noising and Backward Denoising. Basically, noise is continuously added to the image, and the challenge is to reverse this process to recover the original image. (the u-net model usually used to predict the noise or velocity)

## Forward Diffusion Process
![Forward](https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/forward_equation.png)
During the Forward Diffusion phase, a Markov chain is defined, where each timestep where each timestep t only depends on the previous timestep t−1. We use a variance schedule β to control the mean and variance, with β₀ < β₁< … < βt. We begin at X0, which is sampled from the real data distribution q(x), and iteratively adjust the mean and variance to generate X1, and so on, until reaching the final state XT, which is Gaussian noise. This process can be thought of as gradually pushing the image away from the real data distribution until it becomes indistinguishable from noise.
![Forward](https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/Diffusion_process.png)

## Backward Denoising Process


![training_loss](https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/training_loss.jpg)



-   Image Compression
-   Denoising
-   Image Generation

