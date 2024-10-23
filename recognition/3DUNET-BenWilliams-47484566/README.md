# 3D U-Net
## Model description
The 3D U-Net is an extension of the original 2D U-Net architecture which deals in 3D data instead.
The main focus is that the model performs voxel-wise segmentation where each voxel is assigned a class label. Then the model should be able to accurately determine if a label fits an image
This model is typically used in medical imaging to segment organs or identify tumors in MRI scans. 

## How it works

!Image Here

This image shows the architecture of the U-Net in the U shape. 
The model takes a 3D volumetric image as an input. In this case i was working with grayscale images so 
there is 1 in channel. The model outputs a 3D segmentation map where each 3D pixel is assigned a label for a specific class.
In this case there are 6 classes, so 6 out channels.  
> Like the standard u-net, it has an
> analysis and a synthesis path each with four resolution steps. In the analysis
> path, each layer contains two 3 × 3 × 3 convolutions each followed by a rectified
> linear unit (ReLu), and then a 2 × 2 × 2 max pooling with strides of two in
> each dimension. In the synthesis path, each layer consists of an upconvolution
> of 2 × 2 × 2 by strides of two in each dimension, followed by two 3 × 3 × 3
> convolutions each followed by a ReLu. Shortcut connections from layers of equal
> resolution in the analysis path provide the essential high-resolution features to the synthesis path. [1]

The data that was used in my model is the Labelled weekly MR images of the male pelvis. [3] 
Data was read from nifti file format and loaded into the preferred spacial dimensions.

>  The network output and the ground truth labels are compared
> using softmax with weighted cross-entropy loss, where we reduce weights for the
> frequently seen background and increase weights for the inner tubule to reach
> a balanced influence of tubule and background voxels on the loss.[1]

## Results
After training each of the 6 labels produced a dice coefficient score of ... respectively
The loss, calcuted using Weighted cross entropy loss as per the original model, can be seen as so:


### Reproducability
No specific hardware is required, however a cuda CPU is recommneded for training

### Dependencies

The following dependencies are required to run this project:

- **Python**: 3.8 or later
- **PyTorch**: 1.10.0
- **torchvision**: 0.11.1
- **numpy**: 1.21.0
- **scipy**: 1.7.1
- **scikit-learn**: 0.24.2
- **matplotlib**: 3.4.3
- **tqdm**: 4.62.3

 ## Preprocessing
 1. Normalization:
  Each input MRI scan was normalized to have a mean of 0 and a standard deviation of 1. This was done to ensure that the input data is centered and scaled, which can help speed up the convergence of the model.
2. Resampling:
  The input images were resampled to a consistent voxel size (e.g., 1 mm³) to ensure uniformity across the dataset. This step is crucial in medical imaging as different scanners may produce images with varying resolutions.
3. Padding:
  Images were padded to a uniform size (e.g., 256x256x128) to maintain consistency in dimensions when feeding into the 3D U-Net model. This padding prevents issues related to varying input sizes, which can lead to errors during batch processing.

## Training Testing splits
1. 70% training split - this split is the biggest because when training, the model would need to know most of the underlying patterns and they can do that with most of the data
2. 15% validation - The validation set is used to tune hyperparameters and choose the best model. It doesnt need to be as big as the training set
3. 15% testing - This final split is used to find out how well the model was trained

### References
1. O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense Vol-
umetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Inter-
vention – MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. Joskowicz, M. R. Sabuncu,
G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp. 424–432.
2. OpenAI. (2023). ChatGPT (Mar 14 version) [Large language model]. https://chat.openai.com/chat

3. https://data.csiro.au/collection/csiro:51392v2?redirected=true

