# Alzheimer’s disease Image Classification Task

This project aims to classify brain images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset into two categories: normal and Alzheimer's disease (AD) using the GFNet vision transformer. The objective is to understand the use of Transformer model and its effectiveness on image classification. Ultimately, I hope to achieve an accuracy close to 0.8 on the test set. The GFNet architecture is selected for its efficiency and effectiveness in handling image classification tasks using Global Filters.

## Problem
Alzheimer's disease is a progressive neurodegenerative disorder that leads to cognitive decline and the dementia symptoms will get worse gradually year by year. Therefore, early and accurate diagnosis is very crucial to identify the cause and utilize effective treatment as early as possible. However, brain images could be complex to analyse and classify using simple models. This project addresses the challenge of classifying brain images to assist in the diagnosis of Alzheimer's disease using state-of-the-art Vision Transformer model - GFNet. This also provides a valuable tool for clinicians for automation and efficiency in medical dialogsis.

## Principle of GFNet

GFNet (Global Filter Network) is a cutting-edge vision transformer model that leverage global filter layers to replace self-attention layers in traditional transformer model [1]. It uses Fourier transforms to handle spatial features effectively by converting it to freqeuncy domains. It therefore learns long-term spatial dependencies in images efficiently. Unlike traditional CNNs, which typically depend on local receptive fields, GFNet applies frequency-domain filtering, performing element-wise multiplication between frequency-domain features and the global filter to capture global relationships across an entire image.  The global filtering mechanism allows the model to focus on both fine details and larger structural patterns in brain scans, making it an ideal choice for medical imaging tasks such as Alzheimer's disease classification. Its architecture is shown in the below figure. The uses of global features and fourier transform in GFNet reduces the number of parameters and result in faster training times and improved generalization which helps in differentiating the subtle differences between normal and AD brain images.

 <img src="images/gfnet.png" alt="description" width="300" height="200">



## Data Pre-processing 

### Dataset
The [ADNI brain dataset](https://adni.loni.usc.edu/) is used. It consists of 2D MRI slices labeled as either normal or Alzheimer's disease. Each 2D image is 256 x 240 pixels and represents one of 20 slices from a MRI scan collection. The dataset available in Rangpur is already divided into train and test sets, with 21520 and 9000 images respectively.  

### Preprocessing
For the pre-processing stage, it is carried out in 'dataset.py'. 
1. **Transformations** on training set to ensure model robustness:
   - **Resize**: All images are resized to 224x224 pixels to match the input size required by the GFNet model.
   - **Data Augmentation**: Random horizontal flipping, random rotations, and color jittering by adjusting brightness are used.
   - **Normalization**: Pixel values are normalized to standardize the input. 

   For the validation and test datasets, only resizing and normalization are applied, as no augmentation and shuffling is necessary for evaluation.

2. **Train-validation split**: Patient-wise splitting approach is used. Data is split based on unique patient IDs so that all images from the same patient do not appear in both sets. This prevents overfitting. The train-validation ratio is 8:2. 

3. **Balanced Sampling**: Below shows the class distribution of the original training dataset which is slightly imbalanced, with more NC samples than AD samples. To balanace the training set, class weights are first calculated based on the frequency of each class in the training set and a weighted random sampler is used to reduce the overrepresented class (NC) and to slightly increase the representation of the underrepresented class (AD) relative to the total. 
```
Class distribution before balancing:
------------------------------------
  Class 'AD' (Index 0): 10400 samples
  Class 'NC' (Index 1): 11120 samples
Total samples: 21520

Class distribution after balancing:
------------------------------------
  Class 'AD' (Index 0): 8563 samples
  Class 'NC' (Index 1): 8637 samples
Total samples: 17200
```


  Pretrain gfnet_h_ti model using Adam: 

  <img src="images/image.png" alt="description" width="300" height="200">
  <img src="images/image-1.png" alt="description" width="300" height="200">
   test_newmodules_h_ti.out
   Early stopping triggered after 23 epochs
   Final Train Loss: 0.0013, Train Acc: 99.99%
   Final Val Loss: 0.3999, Val Acc: 92.29%
   Test Accuracy: 72.19%


   Pretrain gfnet_h_ti model using AdamW: 

   <img src="images/image-9.png" alt="description" width="300" height="200">
   <img src="images/image-10.png" alt="description" width="300" height="200">
   newmod_hti_adamw.out
   Early stopping triggered after 39 epochs
   Final Train Loss: 0.0001, Train Acc: 100.00%
   Final Val Loss: 0.4848, Val Acc: 92.96%
   Test Accuracy: 72.17%

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
- [1] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global filter networks for image classification. Advances in neural information processing systems, 34, 980-993.





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
