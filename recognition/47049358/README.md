---
title: COMP3710 Report
author: "Ryuto Hisamoto"
date: "2024-10-25"
---

# Improved 3D UNet

Improved 3D UNet is capable of producing segmentations for medical images. The report covers the architecture of model, its parameters and relevant components, and its performance on 3D prostate data.

 ## The Problem

 Segmentation is a task that requires a machine learning models to divide image components into meaningful parts. In other words, the model is required to classify components of an image correctly into corresponding labels.

## The Model

<p align="center">
  <img src = documentation/model_architecture.png alt = "Improved 3D UNet Architecture" width = 100% >  
  <br>
  <em>Figure 1: Improved 3D UNet Architecture</em>
<p>
  
 UNet is an architecture for convolutional neural networks specifically for segmentation tasks. The model takes advantage of skip connections and tensor concatenations to preserve input details and its structure while learning appropriate segmentations. The basic structure of UNet involves the downsampling and upsampling of original images with skip connections in between corresponding pair of downsampling and upsampling layers. Skip connection is a technique used to (1) preserve features of the image and (2) prevent diminishing gradients over deep layers of network(Gupta, 2021). The authors of "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge" proposes the improvement on the architecture by the integration of segmentation layers at different levels. The resulting architecture is improved 3D UNet which is capable of performing complex segmentation tasks with appropriate parameters and components.

 In the context pathway (encoding part), 3 x 3 x 3 convolution with a stride and padding of 1 is applied in each convolutional layer. Then, instance normalisation is applied and its output groes through leaky ReLU with a negative slope of $10 ^ {-2}$ as an activation function. We refer to this module as a 'standard module', and 3 x 3 x 3 stride 2 convolution is the same as a standard module except its stride being 2 to reduce the resolution of input.Context modules are composed of two standard modules with a drop out layer in-between with a dropout probability of 30%. This helps in reducing computational cost and memory requirements. Lastly, output from context modules are combined with its input passed from a standard module with element-wise sum. From the 2nd level, the depth of layers is doubled, and the process is repeated throughout each level of the context pathway.

 The localisation pathway (decoding part) utilises a 4 x 4 x 4 transposed convolution with a stride of 2 and padding of 1 to increase the resolution while reducing the feature maps. As the input goes up layers, they are concatenated with the output from a context module on the same layer to preserve features which are potentially lost as they go through the network. Then, localisation modules combines the features together while reducing the number of feature maps to reduce memory consumption. Its output is handed over to the following upsampling module, and the process is repeated until it reaches back to the original level of the architecture. When the input reaches to the original level, it goes through another standard module before handed over to a segmentation layer and is summed with the previsous outputs of segmentation layers.

From the third localisation layer, segmentation layers which apply 1 x 1 x 1 convolution with a stride of 1 take outputs from localisation modules and map them to the corresponding segmentation labels and are summed element-wise after upscaled to match the size. Finally the output is applied a softmax to turn its predictions of labels into probabilities for later calculation of loss. It is to be noted that, argmax has to be applied to produce proper masks from the output of the model. Otherwise, the model produces its predictions from the architecture and processed discussed.

### Loading Data

The authors of "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge" seem to have used the following augmentation methods:

- Random rotation
- Random scaling
- Elastic transformation
- Gamma correction
- Mirroring (assumably horizontal flip given their problem space)

However, some augmentations methods are altered to limit the complexity of solution. For instance, use of elastic transformation was avoided as it could alter the image significantly, causing it to deviate from the actual images the model may find. Moreover, the tuning of such complex method could decrease the maintainability of solutin. Therefore, the project preserved basic augmentation techniques to process the training data. More precisely, techniques used are limited to:

- Random Rotation ($[-0.5, 0.5]$ for all x, y, and z coordinates)
- Random Vertical Flip
- Gaussian Noise ($\mu = 0, \sigma = 0.5$)
- Resizing (down to (128 x 128 x 64))

Resizing is an optional transformation as it is meant to save the memory consumption and increase the speed of training. However, with a limit in memory, this transformation must be applied for the training to happen. The past attempts have shown that the implementation cannot process image size of (256 x 256 x 128) regardless of batch size. In addition, all images are normalised as they are loaded to eliminate difference in intensity scales if there are any. Finally, all voxel values are loaded as `torch.float32` but `torch.uint8` is used for labels to save memory consumption. The labels in the dataset are indexed according to the table below, and are assigned to corresponding layers as binary masks as they are on-hot encoded. For example, segments corresponding to the first type of label appears as 1s in the 0th layer while other parts appear as 0s, and so on.

 |Labels|Segment|
| - | - |
|0| Background |
|1| Body |
|2| Bones |
|3| Bladder |
|4| Rectum |
|5| Prostate |


<p align="center">
  <img src = documentation/example_labels_and_images.png alt = "Example of labels layered on top of images" width = 100% >
  <br>
  <em>Figure 2: Example of labels layered on top of images.</em>
<p>

### Training

- Batch Size: 2
- Number of Epochs: 300
- Learning Rate: $5e ^ {-4}$
- Initial Learning Rate (for lr_scheduler): 0.985
- Weight Decay: $1e ^ {-5}$

The model takes in an raw image as its input, and its goal is to learn the best feature map which ends up being a multi-channel segmentation of the original image. 

#### Loss Function

The model utilises dice loss as its loss function. Moreover, it is capable of using deviations of dice loss such as a sum of dice loss and cross-entropy loss, or focal loss. A vanilla dice score has formula: $$D(y_{true}, y_{pred}) = 2 \times \frac{\Sigma(y_{true} \cdot y_{pred})}{\Sigma y_{true} + \Sigma y_{pred}}$$

in which $y_{true}$ is the ground truth probability and $y_{pred}$ is the predicted probability. Hence dice loss is provided by:

$$L_{Dice} = 1 - D(y_{true}, y_{pred})$$

The loss function mitigates the problem with other loss functions such as a cross-entropy loss which tend to be biased toward a dominant class. The design of dice loss provides more accurate representation of the model's performance in segmentation. In addition `monai` provides an option to exclude background from the calculation of loss, and the model makes use of this option when calculating the loss (background is included when testing).

The author recommends the sum of dice loss and a weighted cross-entropy loss (Yang et al., 2022) for the problem as it seems to produces the most optimal performance. Cross-entropy loss is calculating by:

$$L_{CE} = \frac{1}{N} \Sigma_i - [y_i \times \ln (p_i) + (1 - y_i) \times \ln (1 - p_i)]$$

where $y_i$ is the lebel of sample $i$ and $p_i$ represents the probability of sample $i$ predicted to be positive, and $N$ represents the number of samples. Hence the its wegithed sum with a dice loss can be shown as

$$L_{loss} = L_{Dice} + \alpha L_{CE}$$

"Multi-task thyroid tumor segmentation based on the joint loss function" recommends to set $\alpha = 0.2$, so the report strictly follows it to calculate the weighted loss.

#### Optimiser

**Adam (Adaptive Moment Estimation)** is an optimisation algorithm that boosts the speed of convergence of gradient descent. The optimiser utilises an exponential average of gradients, which allows its efficient and fast pace of convergence. Moreover, the optimiser applies a **$L_2$ regularisation** (aka Tikhonov regularisation) to penalise for the complexity of model. Complexity can be defined as the number of parameters learned from the data, and high complexity is likely to be an indication of overfitting to the training samples. Hence, regularisation is necessary to prevent the model from learning high values of parameters by penalising the model for its complexity, and $L_2$ regularisation is one of the explicit regularisation methods which adds an extra penalty term to the cost function. The parameters learned with such technique can be denoted as

$$\hat{\theta} = \arg \min_\theta \frac{1}{n} ||X\theta - y||^ 2_ 2  + \lambda ||\theta|| ^ 2 _ 2$$

In addition, the model utilises a learning rate scheduler based on the number of epochs, which dynamically changes the learning rate over epochs. This allows the model to start from a large learning rate which evntually settles to a small learning rate for easier convergence. In the implementation, the learnign rate is reduced by $1e ^ {-5}$ over each epoch. 

It is to be noted that mixed precision and gradient accumulation are used to reduce the memory consumption during the training. **Mixed precision** reduces the memory consumption by replaceing value types with `torch.float16` where it can to reduce the space required to perform necessary operations including loss and gradient calculations necessary to train the model. **Gradient accumulation** accumulates the gradients and updates the weights after some training loop.

 ### Testing

The model is tested by measuring its dice scores on the segmentations it produces for unseen images. Although the model outputs softmax values for its predicted segmentations, they are one-hot encoded during the test to maximise the contribution of correct predictions. Dice scores for each label is calculated independently to obtain the accurate performance to analyse the model's weakness and strengths in predicting particular segments for all labels. Then, their averages are taken and are summarised in the bar chart. Moreover, the visualisation of first 9 labels are produces with the actual segmentations for comparison.

## Result

## Discussion

Firstly, there had to be a compromise in maintaining the original resolution of the image given the limiation in resources. The model seem to perform well on downsized images, but without testing it on images with original resolution, its performance on original images can only be estimated. Moreover, the optimality of architecture remain as a question as the model could be potentially simplified to perform the same task without facing issues in its large consumption of computer memory.

Secondly, the project did not incorporate the idea of pateient-level predictions. Despite the model's strong performance, its true robustness to scans taken from new patients must be explored to test its true ability to produce segmentations. In future, the model has to be tested for its capability by training it based on patient-level images.



## Conclusion

## References

1. Gupta, P. (2021, December 17). Understanding Skip Connections in Convolutional Neural Networks using U-Net Architecture. Medium. https://medium.com/@preeti.gupta02.pg/understanding-skip-connections-in-convolutional-neural-networks-using-u-net-architecture-b31d90f9670a

2. Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. (2018). Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. In arXiv. https://arxiv.org/pdf/1802.10508v1

3. Yang, D., Li, Y., & Yu, J. (2022). Multi-task thyroid tumor segmentation based on the joint loss function. Biomedical Signal Processing and Control, 79(2). https://doi.org/10.1016/j.bspc.2022.104249
