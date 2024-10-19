# Vector Quantized Variational Autoencoder
## Introduction
This project aims to create a generative VQVAE model for [HipMRI Study on Prostate Cancer](https://doi.org/10.25919/45t8-p065) using the [processed 2D slices (2D images)](https://filesender.aarnet.edu.au/?s=download&token=76f406fd-f55d-497a-a2ae-48767c8acea2). I achieved over 0.9 Structural Similarity Index Measure with 35 minutes of training on a single GPU.

### Model
VQVAE is a generative model that uses a vector quantization layer to discretize the continuous latent space. The model consists of an encoder, a quantizer, and a decoder. This model uses residual blocks and batch normalization layers to improve the training stability of the model. The encoder and decoder is trained by the reconstruction loss and the commitment loss, while the quantizer is trained using exponential moving average method. I referenced the code from [vvvm23/vqvae-2](https://github.com/vvvm23/vqvae-2/blob/main/vqvae.py), however, the original code was VQVAE2, therefore I modified the structure and simplified the implementation to VQVAE for this specific project.

**ReZero Block**: The ReZero block is a residual block with each block consisting of two convolution and batch normalization layers. The residual connection is added to the output of the second convolution layer. The residual connection is scaled by a learnable parameter and is initialized to zero.

**Residual Stack**: A series of ReZero blocks are stacked together to form a residual stack.

**Encoder**: The encoder maps the input image to a continuous latent space by downsampling the image using convolutional layers and batch normalization. After series of downsampling layers, the feature maps are passed to residual blocks.

**CodeLayer**: The code layer is a vector quantization layer which maps continuous feature vectors into a discrete set of learned embeddings. It maps each feature vector to its nearest embedding based on distance calculations to quantize the continuous input. During the training, the embeddings are updated using an exponential moving average, and a commitment loss is applied to keep the feature vectors close to their assigned embeddings. The code layer outputs the quantized feature vectors, commitment loss and the indices of the embeddings.

**Decoder**: The decoder maps the quantized feature vectors to the original image space by upsampling the feature vectors using transposed convolutional layers and batch normalization. The feature vectors are passed through residual blocks, then upsampled to the original image size.

### Dataset
The data is in nii.gz format, therefore 'nibabel' library is used to convert the data into numpy arrays. The data is then converted to PIL Image format as the torchvision transforms require the input to be in PIL Image format or Tensor. Augmentations are applied to the data to increase the diversity of the data. In every training I conducted, cropping the image to 256x128 because the raw data consists images with 256x128 and 256x144. The data is then transformed to Tensor and normalized to have a mean of 0.5 and a standard deviation of 0.5.

### Training
`train.py` is the main script for training the model. The script reads the configuration file, and initializes the model, train and validation dataloaders, optimizer (Adam). Configuration file is saved to the logs. At the specified frequency, training and validation are conducted, and each reconstruction loss, commitment loss, the total loss and SSIM score are logged. Original and reconstructed train and validation images are saved at a specified frequency as well. After the training, the metrics plot is saved which shows the training and validation losses and SSIM scores at each epoch. The model is also saved. The overall training time is logged.

### Inference
`predict.py` is the main script for inference. The script reads the configuration file, and initializes the model and test dataloader. The model is loaded from the saved model. The script conducts inference on the test data, and saves the original and reconstructed images of the first six samples. The histogram of loss and SSIM score are saved and average loss and SSIM score are logged.

## Hyperparameters
### Training
- **model_parameters**: The parameters for the model in a dictionary format.
    - **in_channels**: The number of input channels.
    - **hidden_channels**: The number of hidden channels in the encoder and decoder.
    - **res_channels**: The number of hidden channels in the residual blocks.
    - **nb_res_layers**: The number of residual blocks in the encoder and decoder.
    - **embed_dim**: The dimension of the embeddings.
    - **nb_entries**: The number of embeddings.
    - **downscale_factor**: The downscale factor of the encoder.
- **logs_root**: The root directory for the logs.
- **log_dir_name**: The name of the log directory. If empty, the current time will be used.
- **log_frequency**: The frequency of logging the training and validation metrics.
- **image_frequency**: The frequency of saving the original and reconstructed images.
- **batch_size**: The batch size.
- **num_epochs**: The number of epochs.
- **learning_rate**: The learning rate.
- **weight_decay**: The weight decay.
- **train_dataset_dir**: The directory of the training dataset.
- **val_dataset_dir**: The directory of the validation dataset.
- **num_samples**: The number of samples to use. If empty, all samples will be used.
- **train_transforms**: The list of augmentations for the training data.
- **val_transforms**: The list of augmentations for the validation data.

### Inference
- **model_parameters**: The parameters for the model in a dictionary format.
    - **in_channels**: The number of input channels.
    - **hidden_channels**: The number of hidden channels in the encoder and decoder.
    - **res_channels**: The number of hidden channels in the residual blocks.
    - **nb_res_layers**: The number of residual blocks in the encoder and decoder.
    - **embed_dim**: The dimension of the embeddings.
    - **nb_entries**: The number of embeddings.
    - **downscale_factor**: The downscale factor of the encoder.
- **pretrained_path**: The path to the saved model.
- **logs_root**: The root directory for the logs.
- **log_dir_name**: The name of the log directory. If empty, the current time will be used.
- **test_dataset_dir**: The directory of the test dataset.
- **num_samples**: The number of samples to use. If empty, all samples will be used.
- **test_transforms**: The list of augmentations for the test data.

## Requirements
- PyYAML
- nibabel
- numpy
- pillow
- scikit-image
- scikit-learn
- scipy
- torch
- torchvision

The packages can be installed using the following command:
```bash
pip install -r recognition/VQVAE_47323254/requirements.txt
```

## Execution
### Training
Configure the training parameters in `recognition/VQVAE_47323254/configs/train.yaml` and run the following command:
```bash
python recognition/VQVAE_47323254/src/train.py --config recognition/VQVAE_47323254/configs/train.yaml
```

### Inference
Configure the inference parameters in `recognition/VQVAE_47323254/configs/test.yaml` and run the following command:
```bash
python recognition/VQVAE_47323254/src/predict.py --config recognition/VQVAE_47323254/configs/test.yaml
```

## Results

## References