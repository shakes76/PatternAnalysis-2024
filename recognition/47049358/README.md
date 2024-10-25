---
title: COMP3710 Report
author: "Ryuto Hisamoto"
date: "2024-10-25"
---

# Improved 3D UNet

Improved 3D UNet is capable of producing segmentations for medical images. The report covers the architecture of model,its parameters and relevant components, and its performance on 3D prostate data.

## The Problem

Segmentation is a task that requires a machine learning models to divide image components into meaningful parts.
In other words, the model is required to classify components of an image correctly into corresponding labels.

## The Model

<p align="center">
  <img src = documentation/model_architecture.png alt = "Improved 3D UNet Architecture" width = 100% >  
  <br>
  <em>Figure 1: Improved 3D UNet Architecture</em>
<p>
  
UNet is an architecture for convolutional neural networks specifically for segmentation tasks (Gupta, 2021).
The model takes advantage of skip connections and tensor concatenations to preserve input details and its structure while learning appropriate segmentations.
The basic structure of UNet involves the downsampling and upsampling of original images with skip connections in between corresponding pair of downsampling and upsampling layers.
Skip connection is a technique used to (1) preserve features of the image and (2) prevent diminishing gradients over deep layers of network preventing the learning of parameters (PATHAK, 2024).
The authors of "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge" proposes the improvement on the architecture by the integration of segmentation layers at different levels. The resulting architecture is improved 3D UNet which is capable of performing complex segmentation tasks with appropriate parameters and components.

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

It is recommended to use the sum of dice loss and a weighted cross-entropy loss (Yang et al., 2022) for the problem as it seems to optimise the performance the most. Cross-entropy loss is calculating by:

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

The outcome shows the significant impact of the choice of loss function in the performance of model. It was found that with other loss functions, the model performs poorly on assigning correct labels to small segments. Specifically, segment label 4 (rectum) often suffered from poor performance as it was often ignored by the model in optimising the segmentaion of corresponding label. However, the addition of weighted cross-entropy loss seem to enforce the model to classify segments correctly, which might have led to an improvement in performance. 

## Discussion

Firstly, there had to be a compromise in maintaining the original resolution of the image given the limiation in resources. The model seem to perform well on downsized images, but without testing it on images with original resolution, its performance on original images can only be estimated. Moreover, the optimality of architecture remain as a question as the model could be potentially simplified to perform the same task without facing issues in its large consumption of computer memory.

Secondly, the project did not incorporate the idea of pateient-level predictions. Despite the model's strong performance, its true robustness to scans taken from new patients must be explored to test its true ability to produce segmentations. In future, the model has to be tested for its capability by training it based on patient-level images.

Finally, although the report strictly followed the implementation of the architectures and loss functions from the published papers with different problem space, there could be more optimal or efficient adjustments that could improve the model's performance in terms of accuracy and time and/or memory savings. Therefore, future research
could focus on the improvement of current model with differentiations from the architectures and components already mentioned by researchers for new discoveries.

## Conclusion

Improved 3D UNet is a powerful architecture which makes complex image-processing tasks possible. However, its performance is truly maximised through the observation of its behaviour and performance under different settings, tunings, and/or parameter selections. In the given problem of segmenting 3D prostate images, adjusting the loss function from a vanilla dice loss to the sum of dice loss and weighted cross-entropy loss improved the performance dramatically. The model could be explored in depth in regards to its relationship with its components for improved performance, which could potentially lead to a discovery of new and more generalised architectures that could function in wider 

## References

1. Gupta, P. (2021, December 17). Understanding Skip Connections in Convolutional Neural Networks using U-Net Architecture. Medium. https://medium.com/@preeti.gupta02.pg/understanding-skip-connections-in-convolutional-neural-networks-using-u-net-architecture-b31d90f9670a

2. Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. (2018). Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. In arXiv. https://arxiv.org/pdf/1802.10508v1

3. PATHAK, H. (2024, July 21). How do skip connections impact the training process of neural networks? Medium. https://medium.com/@harshnpathak/how-do-skip-connections-impact-the-training-process-of-neural-networks-bccca6efb2eb

4. Yang, D., Li, Y., & Yu, J. (2022). Multi-task thyroid tumor segmentation based on the joint loss function. Biomedical Signal Processing and Control, 79(2). https://doi.org/10.1016/j.bspc.2022.104249

## Dependencies

# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
_libgcc_mutex=0.1=main
_openmp_mutex=5.1=1_gnu
blas=1.0=mkl
brotli-python=1.0.9=py312h6a678d5_8
bzip2=1.0.8=h5eee18b_6
ca-certificates=2024.9.24=h06a4308_0
certifi=2024.8.30=py312h06a4308_0
charset-normalizer=3.3.2=pyhd3eb1b0_0
contourpy=1.3.0=pypi_0
cuda-cudart=11.8.89=0
cuda-cupti=11.8.87=0
cuda-libraries=11.8.0=0
cuda-nvrtc=11.8.89=0
cuda-nvtx=11.8.86=0
cuda-runtime=11.8.0=0
cuda-version=12.6=3
cycler=0.12.1=pypi_0
expat=2.6.3=h6a678d5_0
ffmpeg=4.3=hf484d3e_0
filelock=3.13.1=py312h06a4308_0
fonttools=4.54.1=pypi_0
freetype=2.12.1=h4a9f257_0
fsspec=2024.10.0=pypi_0
giflib=5.2.2=h5eee18b_0
gmp=6.2.1=h295c915_3
gnutls=3.6.15=he1e5248_0
idna=3.7=py312h06a4308_0
intel-openmp=2023.1.0=hdb19cb5_46306
jinja2=3.1.4=py312h06a4308_0
joblib=1.4.2=pypi_0
jpeg=9e=h5eee18b_3
kiwisolver=1.4.7=pypi_0
lame=3.100=h7b6447c_0
lcms2=2.12=h3be6417_0
ld_impl_linux-64=2.40=h12ee557_0
lerc=3.0=h295c915_0
libcublas=11.11.3.6=0
libcufft=10.9.0.58=0
libcufile=1.11.1.6=0
libcurand=10.3.7.77=0
libcusolver=11.4.1.48=0
libcusparse=11.7.5.86=0
libdeflate=1.17=h5eee18b_1
libffi=3.4.4=h6a678d5_1
libgcc-ng=11.2.0=h1234567_1
libgomp=11.2.0=h1234567_1
libiconv=1.16=h5eee18b_3
libidn2=2.3.4=h5eee18b_0
libjpeg-turbo=2.0.0=h9bf148f_0
libnpp=11.8.0.86=0
libnvjpeg=11.9.0.86=0
libpng=1.6.39=h5eee18b_0
libstdcxx-ng=11.2.0=h1234567_1
libtasn1=4.19.0=h5eee18b_0
libtiff=4.5.1=h6a678d5_0
libunistring=0.9.10=h27cfd23_0
libuuid=1.41.5=h5eee18b_0
libwebp=1.3.2=h11a3e52_0
libwebp-base=1.3.2=h5eee18b_1
llvm-openmp=14.0.6=h9e868ea_0
lz4-c=1.9.4=h6a678d5_1
markupsafe=2.1.3=py312h5eee18b_0
matplotlib=3.9.2=pypi_0
mkl=2023.1.0=h213fc3f_46344
mkl-service=2.4.0=py312h5eee18b_1
mkl_fft=1.3.10=py312h5eee18b_0
mkl_random=1.2.7=py312h526ad5a_0
monai=1.4.0=pypi_0
mpmath=1.3.0=py312h06a4308_0
ncurses=6.4=h6a678d5_0
nettle=3.7.3=hbbd107a_1
networkx=3.2.1=py312h06a4308_0
nibabel=5.3.2=pypi_0
numpy=1.26.4=pypi_0
openh264=2.1.1=h4ff587b_0
openjpeg=2.5.2=he7f1fd0_0
openssl=3.0.15=h5eee18b_0
packaging=24.1=pypi_0
pillow=10.4.0=py312h5eee18b_0
pip=24.2=py312h06a4308_0
pyparsing=3.2.0=pypi_0
pysocks=1.7.1=py312h06a4308_0
python=3.12.7=h5148396_0
python-dateutil=2.9.0.post0=pypi_0
pytorch=2.5.0=py3.12_cuda11.8_cudnn9.1.0_0
pytorch-cuda=11.8=h7e8668a_6
pytorch-mutex=1.0=cuda
pyyaml=6.0.2=py312h5eee18b_0
readline=8.2=h5eee18b_0
requests=2.32.3=py312h06a4308_0
scikit-learn=1.5.2=pypi_0
scipy=1.14.1=pypi_0
setuptools=75.1.0=py312h06a4308_0
six=1.16.0=pypi_0
sqlite=3.45.3=h5eee18b_0
sympy=1.13.1=pypi_0
tbb=2021.8.0=hdb19cb5_0
threadpoolctl=3.5.0=pypi_0
tk=8.6.14=h39e8969_0
torchaudio=2.5.0=py312_cu118
torchtriton=3.1.0=py312
torchvision=0.20.0=py312_cu118
typing_extensions=4.11.0=py312h06a4308_0
tzdata=2024b=h04d1e81_0
urllib3=2.2.3=py312h06a4308_0
wheel=0.44.0=py312h06a4308_0
xz=5.4.6=h5eee18b_1
yaml=0.2.5=h7b6447c_0
zlib=1.2.13=h5eee18b_1
zstd=1.5.6=hc292b87_0
