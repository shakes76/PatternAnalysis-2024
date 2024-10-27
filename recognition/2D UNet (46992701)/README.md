# Segmentation of Hip MRIs using a 2D UNet
##### Author: Malaika Vaz, 46992701

### Files:
* `dataset.py` contains the dataloader for loading and preprocessing the HIP MRI Data
* `modules.py` contains the 2d UNet model class as well as encoder, decoder and conv_block classes with implement the basic blocks of the UNet network architecture.
* `params.py` contains global variables such as file directories for the data and hyperparameters for training.
* `train.py` contains the code for training, validating, and saving the UNet model, as well as plotting the training losses and validation dice score.
* `predict.py` loads the trained model and visualizes example usage of the trained model and metrics.

## Introduction:
The aim of this project is to segment the HipMRI Study on Prostate Cancer data using the processed 2D slices (2D images) with the 2D UNet and achieve a minimum Dice similarity coefficient of 0.75 on the test set.

## Dataset:
The MRI images used for this project were acquired as part of a retrospective MRI-alone radiation therapy study from the Calvary Mater Newcastle Hospital and are in the NifTI file format. The dataset comprises of 11,460 training images with 11,460 corresponding segmented images, 660 validation images and corresponding segmentations, and 540 testing images and corresponding segmentation images.

The provided validation and test sets were used and so there was no train-validate-test split performed for training, validating and testing the 2D UNet model.

### Data Preprocessing:
Images were read using provided NifTI load_data_2D function. All images where normalized and resized to 256x128 to ensure consistency, and the segmentation labels were one-hot encoded.

### Usage:
To create predictions and evaluate the model, set the values of `TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR` in `params.py` to the appropriate file directories of the corresponding images/masks. Run `train.py` to train the model. If a pre-trained checkpoint is available, set variable `CHECKPOINT_DIR` in `params.py` to the directory of the pre-trained checkpoint, and run `predict.py`.

## Model:
The 2D UNet uses the architecture shown in the diagram below:

![UNet network architecture](readme-images/u-net-architecture.png)
*Figure 1: UNet model architecture*

The UNet is a convolution neural network architecture that consists of an encoder and a decoder, joined by skip connections. The model's contracting and expanding architecture gives it a characteristc U-shape which lends the model its the name.

The encoder (shown on the left side of the figure) is a contracting path that repeatedly applies two 3x3 convolution layers followed by a RELU activation and a 2x2 max pooling operation with a stride of 2 for downsampling. This reduces the spatial dimension of the data and extracts important features. 

The decoder (shown on the right side of the figure) then upsamples the input at each step, followed by a 2x2 convolution, which halves the number of features, and a concatenation with the skip connections to re-introduce the spatial details lost during downsampling. This use of skip connections in the UNet architecture enables a more detailed, accurate segmetation mask to be determined and improves final segmentation results.  


## Training:
The model has been trained for 40 epochs, using an ADAM optimiser with an initial learning rate of 1e-4. The learning rate is reduced by a factor of 0.1 if mean epoch loss plateaus over 2 epochs (i.e. patience = 2). The model uses Cross-Entropy loss and DICE score as the evaluation metrics.

## Testing and Validation:
The model's performance is assessed using a DICE score, evaluated on the validation dataset, which is distinct from the training dataset. The training loss at each iteration, the average training losses per epoch and the dice score for each iteration have been plotted to visualise performance of the 2D UNet model.

## References:
1. O.Ronneberger, P.Fischer, and T.Brox,"U-Net: Convolutional Networks for Biomedical Image Segmentation," in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N.Navab, J.Hornegger, W.M.Wells, and A.F.Frangi,Eds. Cham: Springer International Publishing, 2015, pp.234–241.
2. https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
3. https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114 
4. https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201