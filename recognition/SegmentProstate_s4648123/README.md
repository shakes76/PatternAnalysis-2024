
----
 **Documentation**
1. The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation
2. It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.
3. provide example inputs, outputs and plots of your algorithm
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.

# Using a 3D UNet to segment MR images of the male pelvis
The task is to segment the down-sampled Prostate 3D dataset (Dowling & Greer, 2021) using a 3D U-Net model based on the architecture detailed by Çiçek et al. (2016) in the paper *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*. The objective is to achieve a minimum Dice similarity coefficient of 0.7 for all labels on the test set. There are a total of 6 labels: (1) Background, (2) Body, (3) Bones, (4) Bladder, (5) Rectum and (6) Prostate.

---
### Usage
**Dependencies**
- Monai 1.4.0 (for transformations)
- Pytorch 2.3.0
- Mathplotlib 3.9.2
- nibabel 5.3.2

### The  model
<div align="center">
  <img src="images/unet-architecture.png" >
</div>

### Pre-processing
**Data Augmentation**

Monai's pytorch-based library was leveraged to perform the transformations on dataset. Here's a brief description and justification for each transformation performed on the training set:
1. **RandFlipd**: This random flipping along each spatial axis (x, y, z) with a probability of 0.5 for each axis introduces rotational variations in the data, which helps the model become invariant to orientation changes.
2. **NormalizeIntensityd**: Intensity normalization scales the pixel values of the images between 0-1, ensuring consistency in intensity distribution across images.
3. **RandCropByLabelClassesd**: This function crops random, fixed-sized regions from the image, centering each crop on a specific class. The choice of class center is based on specified ratios for each class, ensuring a balanced sampling across different classes. I used this function to create six sub-samples, each cropped to a size of (96, 96, 48), which is 37.5% of the original sample. The cropping was based on a specified ratio of (1, 2, 3, 4, 4, 4) for the different classes, ensuring that larger label classes like background, body, and bones do not dominate the training data. Below is an example of how it works, where their sub-samples focus on only 1 class:
<div align="center">
  <img src="images/RandCropByLabelClasses_monai.png" >
</div>
source: (MONAI Consortium, 2024)

4. **ToTensor**: This transformation converts the NumPy array to PyTorch tensors.

For validation, NormalizeIntensity is also applied to maintain consistency in intensity scaling. Resized ensures that all validation images are in a standardized shape of (256, 256, 128), which aligns with the input dimensions the model expects, thus ensuring fair and consistent evaluation. Lastly, ToTensord converts validation data into tensors for seamless model compatibility.


---
### Performance


