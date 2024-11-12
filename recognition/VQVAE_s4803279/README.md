# COMP3710 Project Farhaan Rashid s4803279

## Background
The purpose of this of this model is to learn from historical data and be able to create new images through the use of machine learning. The model I am using is a Vector Quantised Variational Auto Encoder (VQVAE2). This model has 3 main parts, and encoder, a latent vector embedding space and a decoder. The training data is processed through the encoder to match the dimensions of the embedding space where a codebook will map each feature to a matching vector and then the decoder will essentially build back up to the original size of the image.

## Purpose
The purpose of a using a generative model in the medical imaging field has one major benefit, this is that there is not a lot of historical data available or due to the ethics, the data may not be accessible. This make the generation of new and 'fake' imaged helpful. Although the images are fake, they can serve as strong learning tools for students and professionals in the field.

## Model Description
The 3 parts of the model are the Encoder, Vector Quantiser and Decoder. The VQVAE2 model is an generative learning model that learns by compressing images and uses a discrete mapping to reconstruct the images as close to the original. This allows it to eventually generate new images by learning how the features are mapped to the latent embedding space.

### Pipeline
<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/vqvae_struct_0.PNG" alt="VQVAE Pipiline">
</p>

Input Image -> Encoder -> Vector Quaniser -> Decoder -> Generated Image

&nbsp;

<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/vqvae2_struct_0.PNG" alt="VQVAE2 Top and Bottom Level Encoding">
</p>

### Encoder
The encoder in the VQ-VAE2 model compresses the input image into a lower-dimensional latent space, capturing essential features while reducing redundancy. This structure functions similarly to a convolutional neural network, where the top and bottom layers are responsible for preserving different levels of detail. The top level retains fine-grained details, while the bottom layer captures coarser, more general image features. By encoding these varying levels of detail, the encoder allows the model to retain a rich, multi-scale feature representation, which supports the decoder in reconstructing detailed images from these compressed embeddings.

### Vector Quantiser
The vector quantiser plays a crucial role by discretizing the latent representations produced by the encoder. This step converts continuous feature embeddings into discrete codes by mapping each encoded feature to the nearest codebook vector. This quantization process introduces a structured, discrete representation of the image, enabling the model to build a more consistent and learnable latent space. By constraining the latent space to a set of fixed vectors, the vector quantiser enhances the generative capability of the model: it allows the decoder to reconstruct images based on a consistent and organized set of features, which aids in producing realistic images with variations learned from the training data.

### Decoder
The decoder reconstructs the compressed, quantized representations from the vector quantiser back into the original image dimensions. Utilizing transposed convolutional layers, the decoder progressively rebuilds the image, starting from the coarser details and then refining finer structures. The dual-layer architecture of the decoder mirrors the encoder, where the top layer focuses on decoding high-level structural information, and the bottom layer reconstructs lower-level details. This dual-path approach allows the decoder to effectively reconstruct the image while preserving key details, leading to high-quality outputs that reflect the model’s learned structure from the latent space.

### Loss
<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/loss_eq.PNG" alt="Loss Equation">
</p>

The VQ-VAE2 model’s loss function consists of three primary components: reconstruction loss, codebook loss, and commitment loss. The reconstruction loss measures the pixel-level difference between the input and reconstructed images, guiding the model to preserve as much of the original image detail as possible. The codebook loss encourages the encoded features to closely match the nearest discrete vector in the codebook, ensuring that the latent representation remains compact and consistent. Finally, the commitment loss penalizes the encoder if its output drifts too far from the discrete codebook values, thereby encouraging tighter alignment between the continuous encoder output and the discrete embeddings. A commitment cost weight of 0.25 balances the strength of this alignment. Together, these loss terms enable the model to compress input images effectively while reconstructing high-quality outputs that align with the learned discrete representation.

### Preprocessing
For this project, the dataset comprised 12,660 medical images, partitioned into 11,460 for training, 660 for validation, and 540 for testing. Each image was resized to a standard 256x128 resolution to ensure uniformity across inputs. Although no data augmentation was applied due to the controlled nature of the medical images, images with dimensions differing from the target size were resized, ensuring that input shape remained consistent. In terms of class imbalance, the dataset does not have explicitly defined classes, as the model is trained on grayscale medical imaging data where each sample represents similar anatomy. This homogeneity helps the model learn relevant feature structures, though it also limits exposure to variations outside this dataset. Overall, these preprocessing steps allowed for consistent input handling and effective training of the VQ-VAE2 model without introducing biases from class imbalances.

## Training
The training set included 11460 images which is about 90.5% of the data. This meant that the bulk of the data was used for training the model and learning the features and improving the vector quantisation.

### Hyperparameter for Training
During training, several hyperparameters were selected to optimize the model's performance:

#### Learning Rate:
Set to 1e-3, this rate balances convergence speed and stability.

#### Batch Size:
Set to 16, allowing effective memory use and stable gradient updates.

#### Number of Epochs:
Although only 10 epochs were used due to time and computational constraints, the model was able to achieve reasonable reconstruction quality. However, future work could involve increasing the number of epochs to see if more training leads to better SSIM and a smoother loss curve.

#### Beta (Commitment Loss Weight):
Set to 0.25, to control how closely the encoder’s output aligns with the quantized embedding. Initial values were selected based on standard practices for VQ-VAE models. After testing with different values, these settings provided the best balance of reconstruction quality and training efficiency. Future work may involve additional fine-tuning to further enhance model generalization.

#### Hidden Dimensions:
Hidden dimensions of [64, 128] were chosen to provide a suitable capacity for capturing both high- and low-level features without overwhelming the memory. These dimensions allow for effective feature extraction in the encoder while keeping model complexity manageable.

#### Number of Embeddings and Embedding Dimensions:
The number of embeddings was set to [256, 256] for the top and bottom levels, ensuring a sufficient codebook size to represent diverse image features while preserving discrete structure. Embedding dimensions of [32, 64] were chosen to balance detail retention with computational efficiency, allowing the model to compress and reconstruct images effectively.

#### Number of Workers:
With num_workers set to 4, the data loading process is optimized, reducing bottlenecks during training.

The training loss: [Training Loss](https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/train_losses.txt)

## Validation
The validation set includes 660 images, representing approximately 5.3% of the dataset. This set provides unseen data during training, helping to monitor the model's performance on data it has not encountered before. By evaluating on the validation set, we gain insights into the model’s generalization ability and fine-tune the model parameters accordingly, reducing the risk of overfitting.

The validation loss: [Validation Loss](https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/val_losses.txt)

## Testing
The testing set included 540 images 4.2% of the data. The testing set is the smallest as the model should already be in a good spot to generate new images by now so the testing set is used to evaluate how well the model works by comparing the generated images with the original ones. This is why the set is tthe smallest.

## Structured Similarity Index Measure
The Structural Similarity Index Measure (SSIM) achieved by the model was 0.8732. SSIM is a perceptual metric that quantifies image quality by comparing structural information between the original and reconstructed images. The closer the SSIM score is to 1, the more structurally similar the images are. In this context, a score of 0.8732 suggests the model preserves a high level of detail and structure from the original medical images, making it effective for generating realistic synthetic images.

The SSIM was only calculated during the testing to check the final reconstruction ability of the trained model.

### The training and validation losses
The training and validation loss curves may appear unusual given the modest number of epochs and a learning rate; however, these settings were help maintain stable and incremental learning in the VQ-VAE2 model, which relies on discrete latent representations. While the final loss values may seem low, the model’s architecture and hyperparameters prioritize structural similarity over pixel accuracy, leading to good perceptual quality in the reconstructed images. This balance, reflected in the SSIM score of 0.8732, suggests the model effectively captures essential image features within a limited number of epochs.

<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/loss_plot.png" alt="Loss Plot">
</p>

The trained model: [Trained Model File](https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/vqvae2_epoch_final.pth)

### Original Image
<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/original_0.png" alt="Original Image">
</p>

### Generated Image
<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/reconstruction_0.png" alt="Generated Image">
</p>

If you have a look at the original and the reconstructed image you can see that the generative model is working very well at learning the generating from the black and white hip MRI data. There are some places where information is lost in the generated model however this is expected ad the reproduction is only 87% accurate.

## Modules
### dataset.py
This where the images are loaded from the Nifti files into pytorch tensors and also checking if the tensors are the correct dimensions ie. [batch, channels, height, width]. After the dataset is loaded the the data loader method is also initialised there.

### modules.py
This file is where the architecture of the VQVAE2 model is stored. This is where the functionality of the Encoder, Vector Quantiser and the Decoder are written the VQVAE2 class brings them all together and can be initialised during the training.

### train.py
This is where the training hyper parameters are set up and the training and validation loop are run. In this file the model is also saved periodically and once the training is finished. The losses are also saved and plotted in here.

### predict.py
This is where the test dataset is loaded and the train model is loaded. The SSIM is also calsulated during the testing loop and then the original and reconstructed images are then saved as png files that can be viewed to see the similaity.

### driver.py
You can run this file as long as all the other files are there to support it and it will run the training loop and the test loop and return the results.

# Version Requirments
Python 3.12.3
&nbsp;

Pytorch 2.4.1
&nbsp;

TQDM 4.66.5
&nbsp;

Matplotlib 3.9.1
&nbsp;

Nibael 5.3.1
&nbsp;

Numpy 1.26.4
&nbsp;

SkImage 0.24.0
&nbsp;


## References

VQVAE Paper - Oord, A. V. D., Vinyals, O., & Kavukcuoglu, K. (2018). Neural discrete representation learning. DeepMind. Retrieved from https://arxiv.org/abs/1711.00937v2

VQVAE2 Paper - Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating diverse high-fidelity images with VQ-VAE-2. DeepMind. Retrieved from https://arxiv.org/abs/1906.00446

CSIRO Dataset - Dowling, Jason; & Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data Collection. https://doi.org/10.25919/45t8-p065
