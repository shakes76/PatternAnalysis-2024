# Segmentation of Hip MRIs using a 2D UNet
##### Author: Malaika Vaz, 46992701

### Files:
* `dataset.py` contains the dataloader for loading and preprocessing the HIP MRI Data
* `modules.py` contains the 2d UNet model class as well as encoder, decoder and conv_block classes with implement the basic blocks of the UNet network architecture.
* `params.py` contains global variables such as file directories for the data and hyperparameters for training.
* `train.py` contains the code for training, validating, and saving the UNet model, as well as plotting the training losses and validation dice score.
* `predict.py` loads the trained model and visualizes example usage of the trained model and metrics.

## Introduction:
The aim of this project is to segment the HipMRI Study on Prostate Cancer data using the processed 2D slices (2D images) with the 2D UNet [^1] and achieve a minimum Dice similarity coefficient of 0.75 on the test set.

## Dataset:
The MRI images used for this project were acquired as part of a retrospective MRI-alone radiation therapy study from the Calvary Mater Newcastle Hospital and are in the NifTI file format. The dataset comprises of 11,460 training images with 11,460 corresponding segmented images, 660 validation images and corresponding segmentations, and 540 testing images and corresponding segmentation images.

The provided validation and test sets were used and so there was no train-validate-test split performed for training, validating and testing the 2D UNet model.

### Data Preprocessing:
Images were read using PIL and converted to tensors with lesion images in RGB format and masks as greyscale. All images where normalized and resized to 128x128 to reduce training time and memory consumption.




## References:
[^1]: O.Ronneberger, P.Fischer, and T.Brox,"U-Net: Convolutional Networks for Biomedical Image Segmentation," in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N.Navab, J.Hornegger, W.M.Wells, and A.F.Frangi,Eds. Cham: Springer International Publishing, 2015, pp.234–241.
[^2]: https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
[^3]: https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114 
[^4]: https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201