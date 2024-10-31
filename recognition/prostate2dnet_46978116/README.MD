# Segmenting HipMRI Study for prostate cancer radiotherapy Data 

In this report, we will attempt to segment HipMRI data based on the part of the body the image depicts with the help of 2D UNet Convolutional Neural Network. 

# Understanding the Dataset

The dataset consists 2D MRI slices of the male pelvis collected from a radiation therapy study at Calvary Mater Newcastle Hospital. Each MRI image has a corresponding segmentation image, which we will refer to as the mask, that segments the MRI image based on the part of the body depicted in each segment. There are six classes to these segments: the background, bladder, body, bone, rectum and prostate. Each image is roughly 256 x 128 pixels although some a slightly larger than this. Our task is to develop a method that accurately segments MRI data of the male pelvis based on the training data collected from these images and their masks. 

# The 2D Unet Model

The chosen model to learn this segmentation is a 2D UNet. The 2D UNet a deep neurel network developed by Ronneburg, etal in 2015. This model shows promising performance in segmentation of medical data, making it an ideal fit for the task. The 2D UNet utilisizes an encoder-decoder structure with skip connections. This makes it effective in computervision segmentation problems as it can capture the spatial information of the data well and retains information through the skip connections between the encoder and decoder layers. The model also trains quickly with higher accuracy.

The architecture consists of two 2D convolutional layers with corresponding max pooling layers each. The activation function is ReLu and 2D Batch normalization is used as well. Although Batch normalizaiton was not used in the original paper (because it wasn't developed yet) we have used it here to improve performance. The data is downsampled using these blocks until it reach 512 where there is a bottleneck. THen it is upscaled back to original image size with skip connections to retain information while reaching a detailed resolution. The final layer has 6 outchannels which corresponds with the 6 classes of segmentation


<image 2d-Unet-architecture.png>

# Output and Results

After training on the test set the best performing model achieved dice co-efficients on the test set of:



for the respective classes. 

Below are the plots of training performance :

< image trainingperformance>

And the predictions made by the model.

Notably, the model slightly underperformed in segmenting for class 4 and 5, the rectum and prostate. Thes classes were identified early on as having poor performance compared to other classes which can attributed to their very low prevelance rate. Most the MRI slides do not contain 





