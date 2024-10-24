# Recognition Tasks
# Prostate 3D data Segmentation with Improved 3D-UNet models
## Student information
**Name:** Zheng Yao

**Student number:** s47697106

**Task:** Task #4 

## Introduction:

The U-Net architecture, initially proposed by Ronneberger et al. [5], is a CNN designed for biomedical image segmentation. 
It comprises an encoder-decoder structure with skip connections that enable precise localization and context capture. 
The 3D U-Net extends this concept to volumetric data, making it suitable for 3D medical images.

My implementation builds upon this foundation, incorporating modifications inspired by Isensee, F(2018) and Dai, W.(2021) 
to enhance segmentation performance by using methods including Data Augmentation, Instance Normalization and Leaky ReLU Activation Functions. 

## Architecture Description:

The model accepts a 3D input volume and outputs a segmentation map of the same spatial dimensions. 
The network consists of an encoder (contracting path) and a decoder (expanding path), connected via skip connections at 
each corresponding level.

#### Encoder

The encoder comprises four blocks, each containing two convolutional layers followed by a Leaky ReLU activation and 
instance normalization. Each block is followed by a 3D max pooling layer that reduces the spatial dimensions 
by a factor of two.

##### Block 1

Input Layer: Accepts the picture.

Convolutional Layers: Two layers with 16 filters of size 3×3×3, padding='same', and no bias.

Instance Normalization: Applied after each convolution.

Activation: Leaky ReLU with α=0.01.

Pooling: MaxPooling3D with pool size 2×2×2.

##### Blocks 2 to 4

Convolutional Layers: The number of filters doubles with each block (32, 64, 128).

Instance Normalization and Activation: Same as Block 1.

Pooling: MaxPooling3D with pool size 2×2×2.

### Bottleneck

The bottleneck layer bridges the encoder and decoder.

Convolutional Layers: Two layers with 256 filters.

Instance Normalization and Activation: Same as previous layers.

### Decoder (Expanding Path)

The decoder is the same as the encoder's structure but replaces pooling with upsampling using the transposed convolutions.

#### Up Block 1

Transposed Convolution: Upsamples the feature map by a factor of 2 using 128 filters.

Skip Connection: Concatenated with the output from Encoder Block 4.

Convolutional Layers: Two layers with 128 filters.

Instance Normalization and Activation: Applied after each convolution.

#### Up Blocks 2 to 4

Transposed Convolution: Filters decrease (64, 32, 16).

Skip Connections: Concatenated with corresponding encoder outputs.

Convolutional Layers: Two layers per block, filters match the transposed convolution.

Instance Normalization and Activation: Same as Up Block 1.

#### Output Layer

Convolutional Layer: A 1×1×1 convolution with num_classes filters.

Activation: Uses soft max function.

## Improvements to the Model: 

### Data Augmentation

Augmentation helps prevent overfitting by exposing the model to a wider variety of data, improving its ability to generalize
to unseen images. In my implementation I set a 50% chance to randomly flip the picture and a 50% chance to randomly rotate 
the picture. 

This is the first method that I implemented, and it increased the performance of the model dramatically. Before using 
this method, the average dice score after 50 epoches are around 5 to 6. However, after the implementation, it brought the 
dice score up to around 7. 

### Instance Normalization

According to Ulyanov, D.(2016), instance normalization normalizes the input across each instance (sample) and channel,
rather than across the batch as in batch normalization. It reduces internal covariance shift and stabilizes training, 
especially beneficial when batch sizes are small—a common scenario in 3D medical image processing due to memory constraints. 

I implemented this together with the Leaky ReLU Activation Functions, but did not see too much change when training the
model. 

### Leaky ReLU Activation Functions

The function is defined as: when x >=0, it returns x and when x < 0 , it returns ax where a is a small positive constant. 
By allowing a small gradient when inputs are negative, it helps maintain active gradients during backpropagation, improving learning.

## Model training and Performance: 

### Training data

Training Set: 72% of the total data.

Validation Set: 8% of the total data (10% of the training set).

Test Set: 20% of the total data.

#### Training parameters

Optimizer: Adam optimizer with a learning rate of 10^-4

Batch Size: 2.

Epochs: 100.

Model Checkpoint: Saves the best model based on validation loss.

Early Stopping: Stops training if validation loss does not improve for 10 consecutive epochs.
early stopping was never triggered after I added the Leaky ReLU Activation Functions. 

### training results
Here is a picture of the accuracy rate and the lost. As can be seen from the two graphs, model 
improved greatly in the first 30 epoches and kept quite steady till around 70 epoches where it 
saw a slite fluctuation. 
![accuracy_plot.png](3D_improved%20U_Net_Zheng_Yao_s4769710%2Fpictures%2Faccuracy_plot.png)
![loss_plot.png](3D_improved%20U_Net_Zheng_Yao_s4769710%2Fpictures%2Floss_plot.png)

### model performance: 
Here is the evaluation of the dice score on the entire data set:

Class 1 Average Dice Score: 0.9809

Class 2 Average Dice Score: 0.9052

Class 3 Average Dice Score: 0.9063

Class 4 Average Dice Score: 0.7574

Class 5 Average Dice Score: 0.7702

Overall Mean Dice Similarity Coefficient: 0.8640

Which I think is a very good result. 

#### Random sample prediction:
I have chosen a random sample from the data set and here is a comparison for the result from my
U Net, the label and the input picture. 

![prediction_example.png](3D_improved%20U_Net_Zheng_Yao_s4769710%2Fpictures%2Fprediction_example.png)

The dice score for this particular data is: 

Class 1 Dice Score: 0.9927937388420105

Class 2 Dice Score: 0.9551582932472229

Class 3 Dice Score: 0.9697940945625305

Class 4 Dice Score: 0.8959497213363647

Class 5 Dice Score: 0.9015402793884277

Mean Dice Similarity Coefficient: 0.9430472254753113

## Reference: 

[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online].
Available: https://arxiv.org/abs/1802.10508v1

[2] W. Dai, B. Woo, S. Liu, M. Marques, C. B. Engstrom, P. B. Greer, S. Crozier, J. A. Dowling, and S. S. Chandras,
“CAN3D: Fast 3D Medical Image Segmentation via Compact Context Aggregation,” arXiv:2109.05443 [cs,
eess], Sep. 2021, arXiv: 2109.05443. [Online]. Available: http://arxiv.org/abs/2109.05443

[3] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. Joskowicz, M. R. Sabuncu,
G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp. 424–432.

[4] Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv preprint arXiv:1607.08022.

[5] dzlab (2021) notebooks/_notebooks /2021-06-13-U_Net_transfer_learning.ipynb https://github.com/dzlab/notebooks/blob/master/_notebooks/2021-06-13-U_Net_transfer_learning.ipynb

[6] DigitalSreeni (2021) 219 - Understanding U-Net architecture and building it from scratch https://youtu.be/GAYJ81M58y8?si=YuUF7ta1zMeeL2AW

[7] Used chatgpt o1 to get some ideas for the code and refine the words for this report. 
