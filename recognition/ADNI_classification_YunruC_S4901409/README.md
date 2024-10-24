# ADNI Classifier
## Introduction
Alzheimer’s disease (AD) affects patients’ memory, behavior and cognition. With no existing drug on the market that can completely cure this disease, early detection and intervention are crucial. Nowadays, various deep learning models have been developed for classifying Alzheimer’s Disease Neuroimaging Initiative (ADNI) data, playing a key role in promoting AD detection. In this study, the Global Filter Neural Network (GFNet), a latest Vision Transformer (ViT), is used to build an ADNI classifier, aiming at achieving at least 0.8 accuracy on the test dataset.

A standard transformer architecture consists of two components: an encoder and a decoder, and it is commonly used for natural language processing tasks. Vision Transformer (ViT) is an example of an encoder-only model, where the transformer is adapted to computer vision. During the ViT processing, the input image is first divided into square patches, flattening into a one-dimensional sequence (Figure 1). These patches are linearly projected to the desired dimensionality (Figure 1). Position embeddings are added to the patches to retain information about their original locations within the image (Albanie, 2023). The resulting sequence is passed through the transformer encoder, and finally, the output is processed by a multilayer perceptron (MLP) to classify the image (Figure 1). 

![image](https://github.com/user-attachments/assets/f7117590-f37c-484b-a087-c0a88def6460)


Figure 1: The Architecture of the Vision Transformer (Bang et al., 2023)


In contrast, the overall architecture of GFNet is similar to the ViT but with some minor changes. The multi-head attention layer in Figure 1 is replaced by a global filter layer as shown in Figure 2. The global filter layer has three key operations: 2D Fourier transform (2D FFT), an element-wise multiplication between frequency domain features and learnable global filters, and a 2D inverse Fourier transform (2D IFFT). When applying GFNet to ADNI data, the 2D FFT breaks down the spatial features (extracted from a brain image) into their frequency components, corresponded to different scales of brain structures (NTi Audio, n.d.). Hence, this transformation enables the model to analyze the global spatial patterns across the brain, which is essential for distinguishing between Alzheimer’s Disease and Normal Control brains. In the frequency domain, the network applies learnable global filters, which are designed to enhance relevant features and suppress irrelevant or noisy components (Zhao et al., 2021). Since global filters operate on a broad scale, they also capture the long-range dependencies, such as the relationship between different brain regions (Zhao et al., 2021), which are crucial for classification tasks. At the end, the 2D IFFT maps the modified features back to the spatial domain, enabling further processing through Feedforward Network (FFN).

<img width="563" alt="image" src="https://github.com/user-attachments/assets/3bd30c3c-aa72-4e6f-8dd6-64db1c9e0cd9">


Figure 2: The Overall Architecture of GFNet (Zhao et al., 2021)


## Description of the Algorithm
The GFNet model is implemented across four Python files: `dataset.py`, `modules.py`, `train.py` and `predict.py`. The purpose of each file and the explanation of the code is described below: 
### 1. Dataset.py



## References
1. Albanie, S. (2023). *Vision Transformer Basics.* YouTube. https://www.youtube.com/watch?v=vsqKGZT8Qn8
2. Bang, J.-H., Park, S.-W., Kim, J.-Y., Park, J., Huh, J.-H., Jung, S.-H., & Sim, C.-B. (2023). *CA-CMT: Coordinate attention for optimizing CMT Networks.* IEEE Access, 11, 76691–76702. https://doi.org/10.1109/access.2023.3297206
3. NTi Audio. (n.d.). *Fast Fourier Transform (FFT): Basics, strengths and limitations.* NTi Audio. https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft
4. Zhao, R., Liu, Z., Lin, J., Lin, Y., Han, S., & Hu, H. (2021). *Global Filter Networks for Image Classification.* arXiv. https://arxiv.org/abs/2107.00645
