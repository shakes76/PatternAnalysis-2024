# GFNet Vision Transformer for Alzheimer's Classification of the ADNI Brain dataset


This Repository contains the code of a pytorch implementation of a custom Global Filter Network (GFnet) based on identifying Alzheimer's disease from 2D MRI scans of the brain. Alzheimer's disease is a progressive neurodegenerative disease that destroys memory and many other important mental functions. Early detection of Alzheimer's is a critical step for providing proper treatment to patients. This project's aim is to address the problem of fast classification of Alzheimer’s Disease from brain scans using the Alzheimer’s Disease Neuroimaging Initiative(ADNI) dataset. The dataset contains a number of sliced MRI brain scan images separated into Cognitive Normal (NC) and Alzheimer's disease(AD) images. This model is built upon from the original GFnet design "Global Filter Networks for Image Classification" [3].

## The Model Architecture
The GFnet is a deep learning architecture originally designed for image classification created by Tsinghua University from their release of "Global Filter Networks for Image Classification" [3]. The GFNet has a transformer-style architecture which replaces the self-attention layer in vision transformers(ViTs) with three main components: 

-   A 2D discrete Fourier transform
-   An element-wise multiplication between frequency-domain features and the learnable global filters
-   A 2D inverse fourier transform

This design was to address the issue of complexity, as the complexity of self-attention and the MLP grows quadratically as an image size increases which results in a model that is hard to scale up for problems(Tsang, 2023). The GFNet's approach is to learn long-term spacial dependencies in the frequency domain with log-linear complexity.

The Model design follows closely to the original implementation shown below:

![Model GFNet](assets/GFnet_intro.gif)

_Image Reference: [Rao, Y., & Zhao, W. (2021). Global Filter Networks for Image Classification. GitHub.](https://github.com/raoyongming/GFNet/tree/master)_


### Why Use GFNet?
The GFNet is designed for image classification and has preformed significantly well on the large visual database ImageNet. For the ADNI dataset the task is very similar, to learn the underlining data structures of the images and to classify a given image to have alzheimer's or not. A major benefit of the GFNet compared to other deep learning options is its scalability which is no doubt an important aspect in the medical research industry. For this problem space we want a fast and accurate solution with the ability of the model to expand to more complex data in the future. 

## About the Dataset
The ADNI dataset has been used in this project to train and test our model. ADNI is a well known Alzheimer's disease research dataset which includes thousands of Magnetic Resonance Imaging (MRI) brain scans. The data has been separated into two groups; Normal Control (NC), which are brain images of heathy individuals, and Alzheimer's Disease (AD), which are individuals which have been diagnosed with Alzheimer disease.

 The ADNI dataset can be download from their website, [ADNI website](https://adni.loni.usc.edu/).

 Here is an example image from the data set:

 ![Example](assets/Example_image.jpeg)

### Pre-processing the Data
The images get pre processed prior to training and testing. This step is completed in the training stage when running train.py which calls dataset.py to process the data. Note: The model assumes there is a training and testing split already in the data directory, this will be expalined under the "Usage" heading. The preprocessing for the training and testing data includes:

 - Splitting the training set to 20% validation, 80% training. This is to help evacuate the models performance.
 - Resizing the images to 256 x 256 pixels to ensure constancy across all images.
 - Setting the Images to gray scale to ensure all images are consistent and adn to reduce computation time. minimal information loss would occur as the images are already presented in a grey scale fashion.
 - Normalizing the images to a mean of 0.1156 and a standard deviation of 0.2202. This was calucted in [utils.py](utils.py) the file by iterating through the training images and averaging their means and standard deviations. This was to help during training.

 **Other preprocessing applied to only the training dataset**:

 - Random Augmentations, random cropping and random horizontal flips. This was to improve the generalized performance of the model.
 


## Usage

    1. Downloading the data structure:

    2. Clone the repository:

    3. Training:

    4. Predictions:

### Requirements

## Results

Figures ect
accuracy


## References

[1] National Institute of Aging. (2023, April 5). Alzheimer’s Disease Fact Sheet. National Institute on Aging.  https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet

[2] Shengjie, Z., Xiang, C., Bohan, R., & Haibo, Y. (2022, August 29). 
3D Global Fourier Network for Alzheimer’s Disease Diagnosis using Structural MRI. MICCAI 2022
Accepted Papers and Reviews. https://conferences.miccai.org/2022/papers/002-Paper1233.html

[3] Rao, Y., & Zhao, W. (2021). Global Filter Networks for Image Classification. GitHub.
https://github.com/raoyongming/GFNet/tree/master

[4] Alzheimer’s Disease Neuroimaging Initiative. (2024). ADNI. https://adni.loni.usc.edu/

[5] Tsang, S.-H. (2023, January 8). Review — GFNet: Global Filter Networks for Image Classification. Medium. https://sh-tsang.medium.com/review-gfnet-global-filter-networks-for-image-classification-6c35c426ab51

‌
‌
