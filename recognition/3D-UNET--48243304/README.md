##3D-UNet for the HipMRI Study on Prostate Cancer

#Project Overview

The goal of the project is to perform segmentation on 3D medical images.it used the dataset of contains MRI scans and corresponding segmentation labels. Training the U-Net model on the scans to learn the segmentation task and then predicting accurate segmentation masks to evaluate.

#dataset Using

The dataset used in the project consists of 3D medical images in nifti format.

reference:Dowling, Jason; & Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data Collection. https://doi.org/10.25919/45t8-p065

#Model Architecture

The project uses a 3D U-Net architecture, the network architecture is :

input channels: 1
Output channels:1
Features:32

The network consists of a contracting path(encoder) that captures context and a expnsive(decoder) path that enable precise localization. it uses binary cross-entropy as the loss function and evaluate performance using the DICE Coefficient, a common metric for segmentation tasks.

How it works:

	Input: 		it input to the model is a 3D MRI volume (include [depth, height, width])
	
	Network architecture: 	it includes encoder and decoder. "The important part of the architecture, which allows us to train on sparse annotations, is the weighted softmax loss function."
	
		encoder: successive convolutional layers followed by max-pooling to downsample the spatial dimension, capturing higher-level features.
		
		decoder: Upsampling layers with skip connections from the encoder layers to recover spatial resolution and maintain feature localization.
	
	Output: 	it outputs a segmentation mask where each voxel is classified as either prostate(1) or non-prostate(0)
	

#Dependences

python			3.8+
pytorch			1.9+
TorchVision		0.10.0+
NumPy			1.19+
nibabel			3.2.1+
tqdm			4.61+
matplotlib		3.4.3



##Reference

1.Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. *arXiv*. https://doi.org/10.48550/arXiv.1606.06650

2.OpenAI. (2024). ChatGPT [Large language model]. https://chatgpt.com
3.Dowling, Jason; & Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data Collection. https://doi.org/10.25919/45t8-p065