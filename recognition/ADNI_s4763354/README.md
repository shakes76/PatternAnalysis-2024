# Alzheimer’s disease Image Classification Task

## Introduction
This project aims to classify brain images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset into two categories: normal and Alzheimer's disease (AD) using the GFNet vision transformer. The objective is to understand the use of Transformer model and its effectiveness on image classification. Ultimately, I hope to achieve an accuracy close to 0.8 on the test set. The GFNet architecture is selected for its efficiency and effectiveness in handling image classification tasks using Global Filters.

## Principle of GFNet

GFNet (Global Filter Network) is a cutting-edge vision transformer model that leverage global filter layers to replace self-attention layers in traditional transformer model. It uses Fourier transforms to handle spatial features effectively by converting it to freqeuncy domains. It therefore learns long-term spatial dependencies in images efficiently. Unlike traditional CNNs, which typically depend on local receptive fields, GFNet applies frequency-domain filtering, performing element-wise multiplication between frequency-domain features and the global filter to capture global relationships across an entire image.  The global filtering mechanism allows the model to focus on both fine details and larger structural patterns in brain scans, making it an ideal choice for medical imaging tasks such as Alzheimer's disease classification. The uses of global features and fourier transform in GFNet reduces the number of parameters and result in faster training times and improved generalization which helps in differentiating the subtle differences between normal and AD brain images.


## Problem
Alzheimer's disease is a progressive neurodegenerative disorder that leads to cognitive decline and the dementia symptoms will get worse gradually year by year. Therefore, early and accurate diagnosis is very crucial to identify the cause and utilize effective treatment as early as possible. However, brain images could be complex to analyse and classify using simple models. This project addresses the challenge of classifying brain images to assist in the diagnosis of Alzheimer's disease using state-of-the-art Vision Transformer model - GFNet. This also provides a valuable tool for clinicians for automation and efficiency in medical dialogsis.


## Methodology
It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.



Two approaches
1. Train from scratch
2. Transfer learning 
### Download Model
Download the pre-trained [GFNet_H_TI](https://drive.google.com/file/d/1Nrq5sfHD9RklCMl6WkcVrAWI5vSVzwSm/view?usp=sharing) and [GFNet_H_B](https://drive.google.com/file/d/1F900_-yPH7GFYfTt60xn4tu5a926DYL0/view?usp=sharing) and place them in the `ADNI_s4763354/` directory.

## Result



  Pretrain gfnet_h_ti model using Adam: 

  <img src="images/image.png" alt="description" width="300" height="200">
  <img src="images/image-1.png" alt="description" width="300" height="200">
   test_newmodules_h_ti.out
   Early stopping triggered after 23 epochs
   Final Train Loss: 0.0013, Train Acc: 99.99%
   Final Val Loss: 0.3999, Val Acc: 92.29%
   Test Accuracy: 72.19%


   Pretrain gfnet_h_ti model using AdamW: 



   Pretrain gfnet_h_b model using Adam optimizer:

   <img src="images/image-5.png" alt="gfnet_h_b model with AdamW" width="300" height="200">
   <img src="images/image-6.png" alt="gfnet_h_b model with AdamW" width="300" height="200">
   new_mod_hb.out
   Early stopping triggered after 25 epochs
   Final Train Loss: 0.0848, Train Acc: 96.63%
   Final Val Loss: 0.2593, Val Acc: 93.91%
   Test Accuracy: 73.49%

   Pretrain gfnet_h_b model using AdamW optimizer:

   <img src="images/image-2.png" alt="gfnet_h_b model with AdamW" width="300" height="200">
   <img src="images/image-3.png" alt="gfnet_h_b model with AdamW" width="300" height="200">

   newmod_hb_adamw.out
   Early stopping triggered after 31 epochs
   Final Train Loss: 0.0616, Train Acc: 97.91%
   Final Val Loss: 0.2892, Val Acc: 93.45%
   Test Accuracy: 73.73%

   Pretrain gfnet_h_b model using AdamW optimizer and Cosine LR Scheduler:

   <img src="images/image-8.png" alt="gfnet_h_b model with AdamW" width="300" height="200">
   <img src="images/image-7.png" alt="gfnet_h_b model with AdamW" width="300" height="200">
   new_mod_hb_adamw_cosine.out:
   Early stopping triggered after 22 epochs
   Final Train Loss: 0.1063, Train Acc: 95.24%
   Final Val Loss: 0.3107, Val Acc: 92.48%
   Test Accuracy: 73.96%




## References
- [1] GFNet GitHub: https://github.com/raoyongming/GFNet.git






Requirements
1. The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
Marking Criteria
1. description and explanation of the working principles of the algorithm implemented and the problem it solves (5 Marks)
2. description of usage and comments throughout scripts (3 Marks)
3. proper formatting using GitHub markdown (2 Mark)
