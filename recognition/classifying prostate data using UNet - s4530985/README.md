# classifying 3d HipMRI study data using Unet
## purpose
We are attempting to segment 2d slices of prostates from the HipMRI dataset.
The data is spereted for us into train, test, and validation sets
## overview of the model
the Core of the Unet architecture is the "skip" connections present between corresponding layers of the encoding and decoding portions of the model
![arcitecture](readMe_images/u-net-architecture.png)
## example usage
The program does not work as of pull request

## dependecies
requirements:
python - <=version 3.7
tqdm - >version 4.66.5
nilearn - >version 0.10.4
pytorch - compute platform = cuda 11.8
torchio - >version 0.20.1

## acknowledgements
code structure based off code found https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet