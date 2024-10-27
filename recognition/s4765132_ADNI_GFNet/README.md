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
This project uses the ADNI dataset obatined from the Alzheimer’s Disease Neuroimaging Initiative [\[2\]](#reference2). This dataset contains two classes: Alzheimer's Disease (AD) and Normal Control (NC) with greyscale images of dimensions 256 × 240. Each class has separate folders for train and test sets. The distribution of the dataset is shown in the table below.

| Dataset | AD | NC |
|---------|----|----|
| Training set | 10400 | 11120 |
| Test set | 4460 | 4540 |

Some examples of the images from dataset is shown below.

![Samples of images](Before_Preprocessing_data_sample.png)

## Setup
- **Programming Language**: Python 3.12.4
- **Dependencies**: 
  - `scikit-learn` 1.5.1
  - `torch` 2.4.0
  - `torchvision` 0.19.0
  - `matplotlib-base` 3.9.2
- **Environment**: Conda 24.5.0

To set up the environment, ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) 

## Data Preprocessing
In the data preprocessing stage, load data first and assign correponding labels to loaded data based on the folder structure. Specifically, AD images are labeled as 0, NC images are labeled as 1. Next, in order to achieve a robust model, the original training set is split into 80% training and 20% validation by using ```train_test_split```. The distribution of data after the split is shown below.

| Dataset | AD | NC |
|---------|----|----|
| Training set | 8320 | 8896 |
| Validation set | 2080 | 2224 |
| Test set | 4460 | 4540 |

To enhance data consistency and improve model convergence, each input image is resized to ```64x64``` pixels, converted into a PyTorch tensor, and normalize images to help stabilize the following training process.

## GFNet Model Building 
The GFNet model has two key components: ```GlobalFilter``` and ```GFNet```. ```GlobalFilter``` module performs frequency-domain filtering and ```GFNet``` is the main neural network architecture designed for image classification. 

The ```GlobalFilter``` module is designed to process input images in the frequency domain. The original code can be found on [GitHub](https://github.com/raoyongming/GFNet). It first uses the 2D FFT to convert the spatial features into the frequency domain for subsequent filtering. Unlike the original implementation, intead of using a fixed filter size, the code in this project made an adjustment for generating the ```complex_weight``` automatically with dimensions based on the transformed shape of ```x``` in the frequency domain. This adjustment is more suitble for this classification task. After frequency filtering, an inverse FFT is used to transfrom features back to the spatial domain.

The ```GFNet``` module is the main neural network model for classification. It consists of convolutional layers, global filter modules, a feed-forward network and a classification layer. 

## Predict

## Model Performance





## References
<a id="reference1"></a>
[1] Rao, Y., Zhao, W., Zhu, Z., Zhou, J., & Lu, J. (2023). GFNet: Global filter networks for visual recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(9), 10960-10973. https://doi.org/10.1109/TPAMI.2023.3263824

<a id="reference2"></a>
[2] ADNI Dataset. Sharing Alzheimer’s Research Data with the World. *Alzheimer’s Disease Neuroimaging Initiative*. https://adni.loni.usc.edu/

<a id="reference3"></a>
[3] Rao, Y., Zhao, W., Zhu, Z., Zhou, J., & Lu, J. (2023). Global filter networks for visual recognition. *GitHub* https://github.com/raoyongming/GFNet
