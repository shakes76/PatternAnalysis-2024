# Vector Quantised AutoEncoder trained over the HipMRI study for Prostate Cancer Radiotherapy
## 1.0 Model Description
A vector quantised variational autoencoder (VQVAE) is an expanded version of the traditional variational autoencoder (VAE) architecture.
The following model was developed in PyTorch by employing an object-oriented programming approach. It features an encoder for deriving a latent representation of a MRI scan, a quantisation layer for embedding the image into a discrete space of embeddings and a decoder for reconstructing a similar image from the stored embeddings. 
Trained over 10,000 MRI scans of the male pelvis provided as a part of the "HipMRI Study for Prostate Cancer Radiotherapy" (Downling & Greer, 2021), it achieves an SSIM score of 0.68 at generating images of various
scans representing different stages of prostrate cancer.
## 2.0 Reasons for Implementing a VQVAE over a VAE
This model was first presented in 2017 through the paper "Neural Discrete Representation Learning" (Oord et al, 2017) and demonstrated a small yet
significant change which involved encoding the inputs as discrete representations rather than continuous ones. For example, when this VQVAE was trained over MRI scans of
the pelvis to reconstruct realistic scans, it encoded each image into a discrete vector stored within a dictionary of embeddings, known as a codebook. Instead of predicting
a normalised distribution of these MRI scans, the model encodes pixels into a categorical distribution of indices in vector space, which are then referenced by the decoder to
reconstruct images. This added layer is helpful for avoiding a problem known as 'posterior collapse' in variational autoencoders (Oord et al, 2017), which occurs when a continuous representation of data in lower dimensional space (i.e. after encoding the input image) proliferates a significant amount of noise - leading to the decoder generating images closer to the mean of the distributed data (Yadav, 2019). A variational autoencoder would be appropriate for datasets such as the MNIST grayscale handwritten digits because this dataset represents a continuous distribution of data. While these MRI scans are also 2D and grayscale in nature, they represent different stages of prostrate cancer, which represents discrete data in form of tumour spread, organs and bone structure. Therefore, a VQVAE is more powerful in this case due to the spatial dimensionality inherent in these MRI scans.
## 3.0 Architecture & Process
![vqvae_architecture drawio](https://github.com/user-attachments/assets/55baf8e7-8fd7-4c85-ac17-030914947c14) 
                        _Figure 1: VQVAE architecture._

Similar to the VAE, an input image (in this case a MRI scan) is provided to the encoder, which maps it to a lower latent representation, known as a feature map. After this step, instead of being passed straight to the decoder, the feature map passes through a quantisation layer, where it is compared against a previously stored dictionary of embeddings. The distance between each encoded vector and embedding is calculated, and the encoded vector is replaced with its nearest embedding. This is referred to as quantisation, and this quantised output is passed to the decoder. The decoder then maps this quantised representation to a high dimensional space and backpropagates the gradients straight to the encoder. 

It is important to note that alongside calculating a reconstruction loss value, the VQVAE calculates an additional three loss values:
- **Codebook Loss**:
A variation of the _k_-means clustering algorithm is employed here, where a group of vector embeddings in the dictionary are treated as clusters and the embedded vector representation is adjusted accordingly. Codebook loss refers to the amount of change in the distance between an embedded vector and the closest mean of cluster of embeddings, undertaken by the VQVAE during quantisation. 
- **Commitment Loss**:
Similar to codebook loss, however instead of moving the embedded vectors closer to the nearest cluster, the clustered embeddings are adjusted according to the embedded vector space provided by the encoder. Otherwise, the dictionary may grow in size and fail to allocate any embeddings to the feature map. This represents a tradeoff within the VQVAE, as both codebook and commitment loss values must be treated equally by the model (Yadav, 2019). The commitment loss calculated by multiplying the commitment cost and the reconstruction loss. When calculating the total training loss, a commitment beta β is also added to scale and specify a level of importance (Oord et al, 2017; Kang, 2024).
- **Perplexity**:
KL Divergence loss is commonly used in VAEs, as demonstrated by Malysheva (2018) and Yadav (2019) in their versions of the VAE. However, perplexity is more relevant for training VQVAEs (Kang, 2024) as it refers to the level of entropy i.e. uncertainty in the embedding space. A low value represents that the embedding space is distributed equally, while a higher value indicates higher entropy, with the VQVAE accessing the embedding space less often. A high perplexity value is typical during the first few epochs when training. Ideally, a lower value is achieved as the model becomes better at generating images.

## 4.0 Instructions for Deployment and Dependencies Required
For the best possible performance, the model needs to be trained using a high-end GPU such as the a100 or p100 using the provided BASH script for the SLURM queue on UQ Rangpur. Nevertheless, it is possible to train this model on consumer GPUs as well, as more recent training was conducted on a Nvidia GeForce RTX 2080 Ti (when experimenting with other learning rates and optimisers; refer to section 8.0). PyTorch libraries were used to build this model, and require the following dependencies to train and run inference properly.

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
### 5.1 ADAM optimiser, learning rate of 2e-4 (Best version of this VQVAE) 
<img width="607" alt="training_loss_values_learning_rate_2e_4" src="https://github.com/user-attachments/assets/61a3967e-b74e-44d5-a7d4-1353fd54f181">

_Figure 2: Loss values for a learning rate of 2e-4, with the ADAM optimiser over 50 epochs_

<img width="626" alt="perplexity_learning_rate_2e_4" src="https://github.com/user-attachments/assets/06b00c4d-f4a4-460f-8ed5-6d4341733fab">

_Figure 3: Perplexity over 50 epochs_

<img width="620" alt="total_loss_learning_rate_2e_4" src="https://github.com/user-attachments/assets/b4129580-7945-46b2-b4fc-ad9f9167afe3">

_Figure 4: Total loss over 50 epochs. Some outliers are encountered at epoch 8 and epoch 30, possibly due to the ADAM optimiser as these peaks are not observed when training using Stochastic Gradient Descent._

### 5.2 ADAM optimiser, learning rate of 2e-3 (Worst version of the VQVAE)

<img width="548" alt="training_loss" src="https://github.com/user-attachments/assets/a45cf15b-f95f-43af-9974-14c05cecc269">

_Figure 5: Loss values for a learning rate of 2e-5, with the ADAM optimiser over 50 epochs. Lowering the number of epochs would be beneficial if training the model again in the future._

<img width="547" alt="perplexity" src="https://github.com/user-attachments/assets/3a1ed759-2932-400c-aeeb-2808f7eba9e0">

_Figure 6: Perplexity over 50 epochs_

<img width="551" alt="total_loss" src="https://github.com/user-attachments/assets/5b1618dc-8c6f-4cc0-a588-3d0e1c263a25">

_Figure 7: Total loss over 50 epochs._

### 5.3 ADAM optimiser, learning rate of 2e-5 (Similar results 5.1, slightly lower loss values)
<img width="547" alt="training_loss" src="https://github.com/user-attachments/assets/4e69cc3c-67f5-4359-999d-64634fcb8e98">

_Figure 8: Loss values for a learning rate of 2e-5, with the ADAM optimiser over 50 epochs_

<img width="562" alt="perplexity" src="https://github.com/user-attachments/assets/969325ed-b6db-4f4d-8827-b8fc8482618c">

_Figure 9: Perplexity over 50 epochs_

<img width="548" alt="total_loss" src="https://github.com/user-attachments/assets/a2a4249b-8553-485c-9f7c-a3bffecccacb">

_Figure 10: Total loss over 50 epochs._

### 5.4 Stochastic Gradient Descent (SGD) optimiser, learning rate of 2e-4
<img width="566" alt="training_loss" src="https://github.com/user-attachments/assets/c7bfbe87-d65d-46e8-a398-c4fe72291990">

_Figure 11: Loss values for a learning rate of 2e-4, with the SGD optimiser over 50 epochs_

<img width="605" alt="perplexity" src="https://github.com/user-attachments/assets/a1a81a3b-33f1-4788-a416-431261ad8b56">

_Figure 12: Perplexity over 50 epochs_

<img width="550" alt="total_loss" src="https://github.com/user-attachments/assets/9cfea4d5-da04-4f14-91f3-12260f3909a4">

_Figure 13: Total loss over 50 epochs_
### 5.5 Stochastic Gradient Descent (SGD) optimiser, learning rate of 2e-5
<img width="547" alt="training_loss" src="https://github.com/user-attachments/assets/a168e55c-ce49-4514-9ba5-a87db6f3d0d6">

_Figure 14: Loss values for a learning rate of 2e-5 with the SGD optimiser._
<img width="552" alt="perplexity" src="https://github.com/user-attachments/assets/d48898a3-3065-49d1-aaaf-6efaba53264c">

_Figure 15: Perplexity for 50 epochs._

<img width="542" alt="total_loss" src="https://github.com/user-attachments/assets/ed543ef6-c4fc-4d8b-8804-a1dbcb76fe3e">

_Figure 16: Total loss for 50 epochs_

## 6.0 Actual vs Reconstructed Scans (can also be found in the recognition folder).
![image](https://github.com/user-attachments/assets/9b454fe0-973e-4d52-9835-026955ebea48)

_Figure 17: Actual Scan._

![image](https://github.com/user-attachments/assets/7746d26c-7739-4325-921b-702cf1eb4734)

_Figure 18: Reconstructed Scan. This scan achieves an SSIM score of 0.68, as demonstrated in the output SLURM script below._


![image](https://github.com/user-attachments/assets/594d9475-aa4c-4208-b437-a8033d15d82d) 

_Figure 19: Actual Scans from the "HipMRI study for Prostate Cancer Radiotherapy" Dataset_

![image](https://github.com/user-attachments/assets/dd24be15-34fd-493a-b5d2-cafd69775159)

_Figure 20: Reconstructed Scans. These scans achieved an SSIM score of 0.68, as demonstrated in the output SLURM script below. This output can be found in the associated folder as well._
![image](https://github.com/user-attachments/assets/6bbdbcb5-6fee-42bf-87bc-efc82b464bb3)

```
commitment loss = 0.012087369337677956, codebook loss = 0.04834947735071182, perplexity = 297.97174072265625
Structural Similarity Index Measure between original and reconstructed images = 0.6682232618331909
```

## 7.0 Pre-Processing
The function provided for retrieving 2D MRI scans stored as NIFTI files was modified after it was discovered that the folders contained images with varying height values. 
The training dataset contained images that varied between 256x128 and 256x144 pixels, and lead to errors when loading these files into memory. As a result, the code now contains a snippet which stores the maximum dimensions found when loading a dataset and adjusts the height/width of the input image accordingly before providing it to the model. All images are also converted to PyTorch tensors before being loaded into memory. 

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

Although these images are greyscale, an extra dimension is added indicating the image channel(s), due to dimensionality errors encountered when training the model. In this case, the number of channels is 1. This was achieved by 'unsqueezing' the tensor to add an extra dimension.

``` images = images.unsqueeze(1)```
``` torch.Size([64, 1, 256, 144]), torch.Size([64, 1, 256, 128])```
## 8.0 Selecting the Hyperparameters, Optimiser and Batch Size
The training, validation and test datasets were already split into different folders on UQ Rangpur as well as the shared link (Chandra, 2024) to be loaded into memory by the model. Initially a batch size of 100 was chosen, however this was later dropped to 64 after there were no observed differences in the loss values and the SSIM between an actual and reconstructed scan. A smaller batch size was also suitable for lower-end consumer GPUs such as the RTX 2080 Ti, as some GPU allocation issues were encountered on UQ Rangpur's a100 and p100 clusters.

According to Malysheva (2018) and Kang (2024) VAE versions, a very small learning rate range between 0.0001-0.0003 (1e-4 to 3e-4) exists for the VQVAE, providing the lowest possible training loss values for 50 epochs with an SSIM score of 0.68. This small learning rate range is supported by Zahin et al (2019), who demonstrate that a learning rate of 1e-4 provided a balance between the convergence rate and a confident final convergence point i.e. the best possible local minimum. Nevertheless, between learning rates 2e-5 and 2e-3 were also used for experimentation, and the training plots are visualised above in section 5.0. A higher learning rate (2e-3) lead to faster convergence, however it is likely that the model reached a saddle point after 8 epochs, because during testing it generated an image filled with noise, attaining a SSIM score of 0.1. In contrast, a lower learning rate (2e-5) did not affect the model significantly, with the SSIM score remaining the same.

ADAM is a profoundly better optimiser for VQVAEs, based on the training. The VQVAE was also optimised using Stochastic Gradient Descent (SGD) for two learning rates (2e-4 and 2e-5), and while this process provided a smoother training process, the moving average for perplexity increased per epoch (refer to section 5.0), which demonstrated that SGD was leading to a noisy embedding space. This became evident when a scan generated by a VQVAE optimised by SGD achieved a SSIM score of 0.52.

![image](https://github.com/user-attachments/assets/4acea2c9-dd03-4c33-900a-bf2269a809d5)

![image](https://github.com/user-attachments/assets/a1b78a14-218c-45e7-b393-dd07e78b394b)

```
commitment loss = 0.012087369337677956, codebook loss = 0.04834947735071182, perplexity = 297.97174072265625
Structural Similarity Index Measure between original and reconstructed images = 0.6682232618331909
```

_Figure 21: Generated scan from a trained VQVAE optimised through the ADAM optimiser (SSIM score: 0.68). This output can be found in the associated folder as well_

<img width="569" alt="reconstructed_scan" src="https://github.com/user-attachments/assets/31ecd58a-b597-4515-bf71-6e27a23d0d53">

![image](https://github.com/user-attachments/assets/432d8885-2a97-41df-8e7e-816a0fa01623)

```
commitment loss = 0.018963614478707314, codebook loss = 0.07585445791482925, perplexity = 322.4786071777344
Structural Similarity Index Measure between original and reconstructed images = 0.5232568979263306
```

_Figure 22: Generated scan from a trained VQVAE optimised through the SGD optimiser (SSIM score: 0.52).This output can be found in the associated folder as well._

## 9.0 Conclusion and Implications
In summary, the VQVAE performs reasonably well at generating MRI scans representing occurences of prostate cancer within the male pelvis. It marginally crosses the required SSIM threshold of 0.6, proving that a generated scan can be utilised for further medical analysis. For future research, the learning rate, optimiser and batch size can be altered further to achieve higher SSIM scores and accuracy. Although the dataset was already distributed evenly between patients, cases and recorded weeks, additional downsampling may also lead to an improved VQVAE.

## 10.0 Bibliography
Chandra, S. (2024). Report: Pattern Recognition, Version 1.57. Retrieved 30th September 2024 from https://learn.uq.edu.au/bbcswebdav/pid-10273751-dt-content-rid-65346599_1/xid-65346599_1

Downling, J. & Greer, P. (2021). _Labelled weekly MR images of the male pelvis_. CSIRO. https://doi.org/10.25919/45t8-p065

Kang. J. (2024, February 15). _Pytorch-VAE-tutorial_. https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

Malysheva, S. (2018, December 28). _Pytorch-VAE_. https://github.com/SashaMalysheva/Pytorch-VAE

Oord, A. v. d., Vinyals, O. & Kavukcuoglu, K. (2017). _Neural Discrete Representation Learning_. https://doi.org/10.48550/arXiv.1711.00937

Yadav, S. (2019, September 1). _Understanding Vector Quantized Variational Autoencoders (VQ-VAE)_ [Blog]. https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a

Zahin, A., Le, T. & Hu, R. (2019). Sensor-Based Human Activity Recognition for Smart Healthcare: Semi-supervised Machine Learning. _Artificial Intelligence for Communications and Networks (pp.450-472)_. 10.1007/978-3-030-22971-9_39.
