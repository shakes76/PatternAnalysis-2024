# Alzheimer’s Disease Classification Using Global Filter Network (GFNet)

## Introduction
This project implements Alzheimer's disease classification using the ADNI dataset and Global Filter Network (GFNet) model.

### GFNet Model
GFNet is a deep learning network, which can efficiently capture long-range dependencies in image data by using frequency-domain analysis. It is especially designed for processing and classifying images. It is originally based on the vision Transformer and CNNs, but introduces some key updates. The self-attention mechanism of the vision Transformer is replaced with Fourier transforms in GFNet, which improve the computational effectiveness and enables more effective global feature extraction [\[1\]](#reference1). Additionally, GFNet incorporates design elements from CNN, especially in its hierarchical model variant, which enables capturing local spatial features. These characteristics make GFNet as a powerful and robust tool for handling complex image classification tasks.  

Here is the architecture of GFNet.
![GFNet Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*rkWAbLHZjMjnOpfmAmFc3w.png)

Based on the architecture of GFNet, there are four key components.
  1. Patch Embedding Layer: The input image is divided into small patches, which are linearly embedded into high-dimensional tokens, creating a set of vectors representing different regions of the image.
  2. Global Filter Layer: This is the core component of GFNet. Based on the vision Transformer design, this layer adds a 2D Fourier Transform (FFT) to convert spatial features into the frequency domain. And then, an inverse Fourier transform (IFFT) is used to transform the results back to the spatial domain. These processes can capture global spatial dependencies and reduce the computational complexity.
  3. Feed Forward Network (FFN): This layer contains layer normalization, a multi-layer perceptron (MLP) and another normalization layer. It ensures that features can propagate smoothly through the network.
  4. Global Avaerage Pooling and Classification Head: After several rounds of the Global Filter layer and FFN, the resulting tokens are aggregated through global average pooling. Finally, through the classification head to get the classification output.

The combination of these componets make GFNet more robust and effective for image classification tasks.

### Dataset 
This project uses ADNI dataset obatined from Alzheimer’s Disease Neuroimaging Initiative [\[2\]](#reference2). This dataset contains two classes: Alzheimer's Disease (AD) and Normal Control (NC) with images of dimension 256 × 240 in greyscale. Each class has separate folders for train and test sets. The distribution of the dataset is shown in the table below.

| Dataset | AD | NC |
|---------|----|----|
| Training set | 10400 | 11120 |
| Test set | 4460 | 4540 |

Some examples of the images from dataset is shown below.
![Samples of images](Before_Preprocessing_data_sample.png)


## Data Preprocessing
In the data preprocessing stage, the original training set is split 80% into training set and 20% into validation set. The distribution of data is shown as below.

| Dataset | AD | NC |
|---------|----|----|
| Training set | 8320 | 8896 |
| Validation set | 2080 | 2224 |
| Test set | 4460 | 4540 |

## Model Building 


## Model Performance



## References
<a id="reference1"></a>
[1] ADNI Dataset. Sharing Alzheimer’s Research Data with the World. *Alzheimer’s Disease Neuroimaging Initiative*. https://adni.loni.usc.edu/

<a id="reference2"></a>
[2] ADNI Dataset. Sharing Alzheimer’s Research Data with the World. *Alzheimer’s Disease Neuroimaging Initiative*. https://adni.loni.usc.edu/
