1. The readme file should contain a title, a description of the algorithm and the problem that it solves
(approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results,
if applicable.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation
and testing splits of the data.

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