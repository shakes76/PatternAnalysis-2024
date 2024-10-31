# Segmenting HipMRI Study for prostate cancer radiotherapy Data 

In this report, we will attempt to segment HipMRI data based on the part of the body the image depicts with the help of 2D UNet Convolutional Neural Network. 

# Understanding the Dataset

The dataset consists 2D MRI slices of the male pelvis collected from a radiation therapy study at Calvary Mater Newcastle Hospital. Each MRI image has a corresponding segmentation image, which we will refer to as the mask, that segments the MRI image based on the part of the body depicted in each segment. There are six classes to these segments: the background, bladder, body, bone, rectum and prostate. Each image is roughly 256 x 128 pixels although some a slightly larger than this. Our task is to develop a method that accurately segments MRI data of the male pelvis based on the training data collected from these images and their masks. 

# The 2D U-Net Model

The chosen model to learn this segmentation is a 2D UNet. The 2D UNet a deep neurel network developed by Ronneburg, etal in 2015. This model shows promising performance in segmentation of medical data, making it an ideal fit for the task. The 2D UNet utilisizes an encoder-decoder structure with skip connections. This makes it effective in computervision segmentation problems as it can capture the spatial information of the data well and retains information through the skip connections between the encoder and decoder layers. The model also trains quickly with high accuracy.

The architecture consists of two 2D convolutional layers which are followed by max pooling layers. The activation function is ReLu and a 2D Batch normalization is used as well. Although Batch normalization was not used in the original paper (because it wasn't developed yet) we have used it here to improve performance. The data is downsampled using these blocks until it reaches a bottleneck at 512 feature channels. Here, it is upscaled back to the original image size using upsampling layers with skip connections to retain information while reaching a detailed resolution. The final layer has 6 outchannels which corresponds with the 6 classes of segmentation


![2D U-Net Architecture](images\2dunet_architecture.png)


# Output and Results

After training on the training set with early stoppage the the dice co-efficients on the test set using cross-entropy loss and minimal transformations:

$$
\begin{table}[h!]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Class} & \textbf{Dice Coefficient (Before)} \\ \hline
0 & 0.9945 \\ \hline
1 & 0.9768 \\ \hline
2 & 0.8689 \\ \hline
3 & 0.6563 \\ \hline
4 & 0.3489 \\ \hline
5 & 0.2344 \\ \hline
\textbf{Average} & \textbf{0.6800} \\ \hline
\end{tabular}
\caption{Dice Coefficients for Each Class Before Applying Transformations and Modifications}
\label{tab:dice_before}
\end{table}
$$



Below are the plots of the training performance:

![Training Performance CrossEntropyLoss](images\graphs\crossplot.png)

And the predictions made by the model:

![Best Predictions [Epoch 17] CrossEntropyLoss (Classes: 0, 1, 2)](images\predictions\bestcross10\epoch_16_classes_0-2.png)

![Best Predictions [Epoch 17] CrossEntropyLoss (Classes: 3, 4, 5)](images\predictions\bestcross10\epoch_16_classes_3-5.png)


with less images for close up:


![Best Predictions Close Up [Epoch 17] CrossEntropyLoss (Classes: 0, 1, 2)](images\predictions\bestcross2\epoch_16_classes_0-2.png)

![Best Predictions Close Up [Epoch 17] CrossEntropyLoss (Classes: 3, 4, 5)](images\predictions\bestcross2\epoch_16_classes_3-5.png)


On the left is the MRI image, the middle is real segment for the corresponding MRI image and the right is the prediction from the model.

Notably, the model underperformed in segmenting for class 3, 4 and 5. These classes were identified early on as having poor performance compared to other classes which can be attributed to their very low prevelance rate. The vast majority of MRI slides do not contain these segments and if they do its very small compared to the other classes. 

To try to combat this transformations were performed on the images to try to improve segmenting of these classes such as rotations and zooming. Furthermore, the loss function was changed to directly include the dice loss (1 - dice coefficient) for each class. This dice loss had an added penalty when the co-efficient was below 0.75 to further incentivize the model to learn how to segment class 4 and 5. Below is the test dice coefficients after these changes, showing an improvement in dice coefficients:

$$
\begin{table}[h!]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Class} & \textbf{Dice Coefficient (After)} \\ \hline
0 & 0.9863 \\ \hline
1 & 0.9777 \\ \hline
2 & 0.8820 \\ \hline
3 & 0.7254 \\ \hline
4 & 0.6833 \\ \hline
5 & 0.8711 \\ \hline
\textbf{Average} & \textbf{0.8543} \\ \hline
\end{tabular}
\caption{Dice Coefficients for Each Class After Applying Transformations and Modifications}
\label{tab:dice_after}
\end{table}
$$

And the plots of training performance also show improvement:

![Training Performance DiceLoss](images\graphs\diceplot.png)

And the predictions made by the model:

![Best Predictions [Epoch 40] DiceLoss (Classes: 0, 1, 2)](images\predictions\bestdice10\epoch_39_classes_0-2.png)

![Best Predictions [Epoch 40] DiceLoss (Classes: 3, 4, 5)](images\predictions\bestdice10\epoch_39_classes_3-5.png)


with less images for close up:

![Best Predictions Close Up [Epoch 40] DiceLoss (Classes: 0, 1, 2)](images\predictions\bestdice2\epoch_39_classes_0-2.png)

![Best Predictions Close Up [Epoch 40] DiceLoss (Classes: 3, 4, 5)](images\predictions\bestdice2\epoch_39_classes_3-5.png)



# Requirements

torch
torchvision
scikit-image
tqdm
nibabel
numpy
albumentatinos
matplotlib


# References

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241. Available: https://doi.org/10.48550/arXiv.1505.04597

[2] J. Schmidt, “Creating and training a U-Net model with PyTorch for 2D & 3D semantic segmentation: Dataset building,” Towards Data Science, Dec. 2, 2020. [Online]. Available: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55.

[3] A. Persson, “PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby,” YouTube, Feb. 23, 2021. [Online]. Available: https://www.youtube.com/watch?v=IHq1t7NxS8k.

[4] Esri, “How U-net works,” ArcGIS API for Python Documentation. [Online]. Available: https://developers.arcgis.com/python/latest/guide/how-unet-works/.

[5] J. Dowling and P. Greer, “Labelled weekly MR images of the male pelvis,” CSIRO Data Access Portal, Sep. 20, 2021. [Online]. Available: https://doi.org/10.25919/45t8-p065.



