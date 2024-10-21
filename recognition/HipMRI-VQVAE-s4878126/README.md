# Vector Quantised AutoEncoder trained over the HipMRI study for Prostate Cancer Radiotherapy
## 1.0 Model Description
A vector quantised variational autoencoder (VQVAE) is an expanded version of the traditional variational autoencoder (VAE) architecture.
The following model was developed in PyTorch by employing an object-oriented programming approach. It features an encoder for deriving a latent representation of a MRI scan, a quantisation layer for embedding the image and a decoder for reconstructing a similar image 
from the stored embeddings. 
Trained over 10,000 MRI scans of the male pelvis provided as a part of the "HipMRI study for Prostate Cancer Radiotherapy", it achieves an SSIM score of 0.67 at generating images of various
scans representing different stages of prostrate cancer.
## 2.0 Reasons for Implementing a VQVAE over a VAE
This model was first presented in 2017 through the paper "Neural Discrete Representation Learning" (Oord et al, 2018) and demonstrated a small yet
significant change which involved encoding the inputs as discrete representations rather than continuous ones. For example, when this VQVAE was trained over MRI scans of
the pelvis to reconstruct realistic scans, it encoded each image into a discrete vector stored within a dictionary of embeddings, known as a codebook. Instead of predicting
a normalised distribution of these MRI scans, the model encodes pixels into a categorical distribution of indices in vector space, which are then referenced by the decoder to
reconstruct images. This added layer is helpful for avoiding a problem known as 'posterior collapse' in variational autoencoders (Yadav, 2019), which occurs when a continuous representation of data in lower dimensional space (i.e. after encoding the input image) proliferates a significant amount of noise - leading to the decoder generating images closer to the mean of the distributed data. If the MNIST dataset containing grayscale handwritten digits was being used to trained, a varitional autoencoder would be appropriate as this represents a continuous set of data. Even if these MRI scans are 2D and grayscale in nature, they represent different stages and spread of prostrate cancer, which represents discrete data, and therefore a VQVAE is more powerful at distinguishion when generating such images.
## 3.0 Architecture & Process
![vqvae_architecture drawio](https://github.com/user-attachments/assets/55baf8e7-8fd7-4c85-ac17-030914947c14) 
                        _Figure 1: VQVAE architecture._

Similar to the VAE, an input image (in this case a MRI scan) is provided to the encoder, which maps it to a lower latent representation, known as a feature map. After this step, instead of being passed straight to the decoder, the feature map passes through a quantisation layer, where it is compared against a previously stored dictionary of embeddings. The distance between each encoded vector and embedding is calculated, and the encoded vector is replaced with its nearest embedding. This is referred to as quantisation, and this quantised output is passed to the decoder. The decoder then maps this quantised representation to a high dimensional space and backpropagates the gradients straight to the encoder. 

It is important to note that alongside calculating a reconstruction loss value, the VQVAE calculates an additional three loss values:
- **Codebook Loss**:
A variation of the _k_-means clustering algorithm is employed here, where a group of vector embeddings in the dictionary are treated as clusters and the embedded vector representation is adjusted accordingly. Codebook loss refers to the amount of change in the interpolation distance between an embedded vector and the closest mean of cluster of embeddings, undertaken by the VQVAE during quantisation.
- **Commitment Loss**:
Similar to codebook loss, however instead of moving the embedded vectors closer to the nearest cluster, the clustered embeddings are adjusted according to the embedded vector space provided by the encoder. Otherwise, the dictionary may grow in size and fail to allocate any embeddings to the feature map. This represents a tradeoff within the VQVAE, as both codebook and commitment loss values must be treated equally by the model (Yadav, 2019). 
- **Perplexity**: 
Refers to the utilisation rate of the dictionary. A low value represents that the embedding space was referred at a low rate by the model, which is typical during the first few epochs when training. Ideally, a higher value is achieved as the model becomes better at generating images.

## 4.0 Instructions for Deployment and Dependencies Required
The model needs to be trained first using the provided SLURM script. PyTorch libraries were used to build this model, and require the following dependencies 
to train and run inference properly.

All modules:
- torch
- torch.nn (for classes and objects such as the VQVAE, Encoder, Quantisation Layer and Decoder)

"train.py":
- torch.optim (for the optimiser, in this case the Adam optimiser is being used).
- torch.utils.data (for dividing the dataset into batches).
- matplotlib (for plotting loss values)

"modules.py"
- torch.nn.functional (for executing activation functions such as ReLU)

"utils.py"
- tqdm (for checking the progress on loading images from the training/testing dataset)
- nibabel (for loading NIFTI scans)
- numpy
- matplotlib (for functions plotting the actual and reconstructed scans)

"dataset.py"
- os (for fetching files from the directory on UQ Rangpur which contains the MRI scans)

"predict.py"
- torchmetrics.image -> StructuralSimilarityIndexMeasure (for calculating the SSIM between actual vs reconstructed scans).
## 5.0 Training Plots

## 6.0 Actual vs Reconstructed Scans (can also be found in the recognition folder).
![image](https://github.com/user-attachments/assets/9b454fe0-973e-4d52-9835-026955ebea48)

_Figure 2: Actual Scan._

![image](https://github.com/user-attachments/assets/7746d26c-7739-4325-921b-702cf1eb4734)

_Figure 3: Reconstructed Scan. This scan achieves an SSIM score of 0.67, as demonstrated in the output SLURM script._

![image](https://github.com/user-attachments/assets/594d9475-aa4c-4208-b437-a8033d15d82d) 

_Figure 4: Actual Scans from the "HipMRI study for Prostate Cancer Radiotherapy" Dataset_

![image](https://github.com/user-attachments/assets/dd24be15-34fd-493a-b5d2-cafd69775159)

_Figure 5: Reconstructed Scans. These scans achieved an SSIM score of 0.67, as demonstrated in the output SLURM script._

## 7.0 Pre-Processing
The function provided for retrieving 2D MRI scans stored as NIFTI files was modified after it was discovered that the files contained images of different dimensions. 
The training dataset contained images that varied between 256x128 and 256x144 pixels, and lead to errors when fetching these files from the cluster. As a result, the code now contains a snippet which stores that maximum dimensions found when loading a dataset and adjusts the height/width of the input image accordingly before providing it to the model. All images are also converted to PyTorch tensors before being loaded into memory. 

```
max_rows, max_cols = 0,0
    for inName in imageNames:
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        if len(inImage.shape) == 3:
         inImage = inImage[: ,: ,0] # sometimes extra dims , remove
        # print(f"{inImage.shape[0]}, {inImage.shape[1]}")
        max_rows = max(max_rows, inImage.shape[0])
        max_cols = max(max_cols, inImage.shape[1])
```

Although these images are grayscale, an extra dimension is added which indicates the channels. In this case, the number of channels is 1. This was achieved by 'unsqueezing' th tensor to add an extra dimension.

``` images = images.unsqueeze(1)```
## 8.0 Splitting the datasets


## 9.0 Bibliography
Chandra, S. (2024). Report: Pattern Recognition, Version 1.57. Retrieved 30th September 2024 from https://learn.uq.edu.au/bbcswebdav/pid-10273751-dt-content-rid-65346599_1/xid-65346599_1

Kang. J. (2024). _Pytorch-VAE-tutorial_. https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

Malysheva, S. (2018). _Pytorch-VAE_. https://github.com/SashaMalysheva/Pytorch-VAE

Oord, A. v. d., Vinyals, O. & Kavukcuoglu, K. _Neural Discrete Representation Learning_. https://doi.org/10.48550/arXiv.1711.00937

Yadav, S. (2019, September 1). _Understanding Vector Quantized Variational Autoencoders (VQ-VAE)_ [Blog]. https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a
