# COMP3710 Project Farhaan Rashid s4803279

## Background
The purpose of this of this model is to learn from historical data and be able to create new images through the use of machine learning. The model I am using is a Vector Quantised Variational Auto Encoder (VQVAE2). This model has 3 main parts, and encoder, a latent vector embedding space and a decoder. The training data is processed through the encoder to match the dimensions of the embedding space where a codebook will map each feature to a matching vector and then the decoder will essentially build back up to the original size of the image.

## Model Description
The 3 parts of the model are the Encoder, Vector Quantiser and Decoder. The VQVAE2 model is an generative learning model that learns by compressing images and uses a discrete mapping to reconstruct the images as close to the original. This allows it to eventually generate new images by learning how the features are mapped to the latent embedding space.

### Pipeline
<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/vqvae_struct_0.PNG" alt="VQVAE Pipiline">
</p>
Input Image -> Encoder -> Vector Quaniser -> Decoder -> Generated Image

<p align="center">
  <img src="https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/vqvae2_struct_0.PNG" alt="VQVAE2 Top and Bottom Level Encoding">
</p>

### Encoder
The encoder is responsible for compressing the input image like a CNN into a smaller feature set. It extracts the important features buy using the top and bottom level encoder architecture where the top level encoder retains the finer details and the bottom level encoder retains the coarse details of the image. The final feature map will be used in the Vector Quantise layer.

### Vector Quaniser
The vector quantiser sits after the encoder and takes the features from the encoder and maps them to some discrete values. This replaces each feature from the encoder to a closely mapped discrete value. This helps the model learn a structured and discrete way of representing continuous information.

### Decoder
The decoder is the final part of the VQVAE pipeline. This work the opposite way of the encoder. The decoder is responsible for taking the output from the vector quantised layer and applying transposed convolutional layers to build them back up to the original image. The decoder also has a top and bottom layer like the enocoder and they funtion the same way where the top layer is used to decode high level features and teh bottom level is to decode lower level features.

### Preprocessing
I did not apply any preprocessing or augment the images as the images were already partitioned into the correct train, validate and test folder. Although I did check that each image was 256x128 and if not then I applied an augmentation to meet the dimensions.

## Training
The training set included 11460 images which is about 90.5% of the data. This meant that the bulk of the data was used for training the model and learning the features and improving the vector quantisation.

The training loss: [Training Loss](https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/train_losses.txt)

## Validation
The validation set included 660 images which is about 5.3% of the data. This ensures that were is some data that the model has not seen yet and can be used to strengthen the training of the model. This improves the generalisation of the model and improves the back propogation.

The validation loss: [Validation Loss](https://github.com/farhaan-r/COMP3710-Project/blob/topic-recognition/recognition/VQVAE_s4803279/Results/val_losses.txt)

## Testing
The testing set included 540 images 4.2% of the data. The testing set is the smallest as the model should already be in a good spot to generate new images by now so the testing set is used to evaluate how well the model works by comparing the generated images with the original ones. This is why the set is tthe smallest.

## Results
This model achieved an average Structures Similarity Index Mearure (SSIM) of 0.8732

The training and validation losses:  
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

# Version Requirments
Python 3.12.3
Pytorch 2.4.1
TQDM 4.66.5
Matplotlib 3.9.1
Nibael 5.3.1
Numpy 1.26.4
SkImage 0.24.0

## References

VQVAE Paper - Oord, A. V. D., Vinyals, O., & Kavukcuoglu, K. (2018). Neural discrete representation learning. DeepMind. Retrieved from https://arxiv.org/abs/1711.00937v2

VQVAE2 Paper - Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating diverse high-fidelity images with VQ-VAE-2. DeepMind. Retrieved from https://arxiv.org/abs/1906.00446

CSIRO Dataset - Dowling, Jason; & Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data Collection. https://doi.org/10.25919/45t8-p065
