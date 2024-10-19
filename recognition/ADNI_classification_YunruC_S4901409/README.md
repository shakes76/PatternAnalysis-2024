# ADNI Classifier - YunruC (S4901409)
## Introduction
Alzheimer’s disease (AD) affects patients’ memory, behavior and cognition. With no existing drug on the market that can completely cure this disease, early detection and intervention are crucial. Nowadays, various deep learning models have been developed for classifying Alzheimer’s Disease Neuroimaging Initiative (ADNI) data, playing a key role in promoting AD detection. In this study, the Global Filter Neural Network (GFNet), a latest Vision Transformer (ViT), is used to build an AD classifier, aiming at achieving at least 0.8 accuracy on the test dataset. 
## Vision Transformer: GFNet
A standard transformer architecture consists of two components: an encoder and a decoder, and it is commonly used for natural language processing tasks. Vision Transformer (ViT) is an example of an encoder-only model, where the transformer is adapted to computer vision. During the ViT processing, the input image is first divided into square patches, flattening into a one-dimensional sequence (Graph 1). These patches are linearly projected to the desired dimensionality (Graph 1). Position embeddings are added to the patches to retain information about their original locations within the image ((https://www.youtube.com/watch?v=vsqKGZT8Qn8). The resulting sequence is passed through the transformer encoder, and finally, the output is processed by a multilayer perceptron (MLP) to classify the image (Graph 1). 

![image](https://github.com/user-attachments/assets/f7117590-f37c-484b-a087-c0a88def6460)
Graph 1: The Architecture of the Vision Transformer (https://github.com/user-attachments/assets/f7117590-f37c-484b-a087-c0a88def6460)



## Description of the Algorithm
## References


