# COMP3710 - Pattern Recognition Report

**Student Number:** 47557648

**Chosen project:** Task 8 - Create a generative model of one of the ADNI brain data set (Diffusion Model)

**Objective:**  Generate a reasonably clear image from the ADNI brain dataset using a diffusion model (The model will be trained conditionally) 

## Background of the Diffusion Model
AI image generation has been a significant topic of discussion in the fields of art and deep learning. Generative models like DALL-E 2 and NovelAI have shown incredible results in the fields of art and computer vision. Several types of generative models already exist, including GANs, VAEs, and flow-based models. Each of these can generate high-quality images, with models like StyleGAN-1 and StyleGAN-2 previously being the state-of-the-art in image generation. However, each model type has its own limitations ~

 > GAN models are potentially unstable training and less diversity in generation due to the adversarial nature. VAE relies on a surrogate loss. Flow models have to design a reversible mapping function between two distribution with jacobian easy to comput.

Diffusion model were first proposed by Sohl-Dickstein et al. (2015) to generate images from noise through a backward denoising process. It provides a slow and iterative process for converting noise into an image, which makes it more scalable than the GAN model. Additionally, since the objective of the diffusion model is to predict the noise at each timestep—a supervised learning task—we can expect its training to be much more stable than that of GANs, which involve unsupervised learning.

## What is Diffusion

 <p align="center">
 <img width="500px" src="https://github.com/Yukino1010/Diffusion_Model/blob/master/image_source/diffusion.png" >
 </p>

The concept of diffusion refers to the movement of substances from a region of higher concentration to a region of lower concentration. Inspired by this concept, diffusion models construct a Markov chain to 'push' or 'pull' the image between the original and target distributions (standard Gaussian distribution) by adding or removing noise from the input data. The final goal of the diffusion model is then to determine the noise added to the image and reverse this process.

The diffusion model consists of two phases: Forward Noising and Backward Denoising. Basically, noise is continuously added to the image, and the challenge is to reverse this process to recover the original image. (the u-net model usually used to predict the noise or velocity)

## Forward Diffusion Process
<p align="center">
  <img width="600px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/forward.png" alt="Backward" />
</p>

During the Forward Diffusion phase, a Markov chain is defined, where each timestep where each timestep t only depends on the previous timestep t−1. We use a variance schedule β to control the mean and variance, with β₀ < β₁< … < βt. We begin at X0, which is sampled from the real data distribution q(x), and iteratively adjust the mean and variance to generate X1, and so on, until reaching the final state XT, which is Gaussian noise. This process can be thought of as gradually pushing the image away from the real data distribution until it becomes indistinguishable from noise.
![Forward](https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/Diffusion_process.png)

## Backward Denoising Process

<p align="center">
  <img width="400px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/backward_1.png" alt="Backward" />
</p>

<p align="center">
  <img width="350px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/backward_2.png" alt="Backward" />
</p>

To remove noise from the noisy image distribution, we need to find an estimated reverse distribution p(x_t-1 | x_t), which is defined as a normal distribution with parameters μ and σ. In the DDPM paper, assuming σ is close to βt, the reverse distribution p(x_t−1|x_t) could be written as:  (the below equation require the number of time step to be large enough e.g. 1000, 2000)

<p align="center">
  <img width="400px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/p(x-1|x).png" alt="Backward" />
</p>

The sampling process of DDPM is defined as follows. This process allows us to remove the noise from the noisy image Xt through an iterative denoising procedure. The only unknown parameter required for the p(x_t−1∣x_t) distribution is Є_θ, which can be estimated by the U-Net model. (In this report I also inplement DDIM, a variation of DDPM which has a faster smapling process)
<p align="center">
  <img width="450px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/DDPM_process.png" alt="Backward" />
</p>

## Training

<p align="center">
  <img width="700px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/training_loss.jpg" alt="Backward" />
</p>

The default setting for the total number of iterations is 10,000, which requires around 5 hours of training on a P100 GPU. However, the graph shows that it converges quickly within just 10,000 iterations. Further adjustments to the number of iterations can be discussed.

The training can be set up by modifying the 'data_dir' variable in the main function with an appropriate data path (in train.py). When you run train.py, the program will automatically train the model and save the result under './results' dir. 

## Requirements
This program is run in the Kaggle environment (4/10/2024) with a P100 GPU. <br> The 'ema-pytorch' package is required to run the program.
## Model Archetecture

<p align="center">
  <img width="700px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/u-net_arc.png" alt="Backward" />
</p>

The model architecture is a standard U-Net model. To generate noise at each corresponding timestep, we usually combine the additional condition t into the U-Net model. Since I am working on a conditional image generation task (label 0: NC, label 1: AD), I will include a second condition, c, to control the model.

## Hyperparameter
- BATCH_SIZE = 8
- LR = 7e-5
- IMG_SIZE = 128
- FILTER_SIZE = 64
- TOTAL_ITERATION = 100000
- SAVE_N_ITERATION = 10000
    
## Loss
<p align="center">
  <img width="600px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/vlb_loss.png" alt="Backward" />
</p>

As the forward and backward diffution process can be written as the joint probability of x from 1 to T, denoted as q(x1:T | x0) and p_θ(x0:T | xT), the objective of the loss function is to make these two distributions as close as possible. To achieve this, we can apply the variational lower bound (VLB) to optimize the negative log-likelihood, -log(p_θ(x)). The objective then becomes minimizing the KL divergence between p and q, which can be simply computed using MSE. (between Є_θ and Є_t)
<p align="center">
  <img width="700px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/loss_fn.png" alt="Backward" />
</p>

## Result
### NC Image Generation
<p align="center">
  <img width="900px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/NC_result_img.jpg" alt="Backward" />
</p>

### AD Image Generation
<p align="center">
  <img width="900px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/training/AD_result_img.jpg" alt="Backward" />
</p>

### Mix Image Generation
<p align="center">
  <img width="900px" src="https://github.com/Yukino1010/PatternAnalysis-2024/blob/topic-recognition/recognition/PO-HSUN_LU_DDIM_47557648/results/result_img10.jpg" alt="Backward" />
</p>

## References
1. ***Denoising Diffusion Probabilistic Models*** [[link](https://arxiv.org/abs/2006.11239)]
2. ***Denoising Diffusion Implicit Models*** [[link](https://arxiv.org/abs/2010.02502)]
3. ***Understanding the Diffusion Model and the theory behind it*** [[link (my article)](https://medium.com/@s125349666/understanding-the-diffusion-model-and-the-theory-tensorflow-cafcd5752b98)]
4. ***Lil' Log - What are Diffusion Models?*** [[link](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)]





