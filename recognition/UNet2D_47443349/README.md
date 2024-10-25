# UNet2D Segmentation of HipMRI Medical Imaging

## Author
Alexander Pitman (47443349)

## Task
The task is to segment 2D medical images from the HipMRI Study on Prostate Cancer. There are six labels: Background, Body, Bone, Bladder, Rectum, Prostate. The goal is to achieve a minimum average Dice Score of 0.75 for each of the six labels on the test set.
Specifically suited for Biomedical Image Segmentation as introduced in the original paper, the UNet architecture was selected to solve this problem.

## Data

### Brief Dataset Description
The dataset contains images and masks which are processed 2D slices from the HipMRI Study on Prostate Cancer, stored in NIfTI format. The images are greyscale and the masks have each pixel assigned an integer value from 0 to 5 representing its label (0 = Background, 1 = Body, 2 = Bone, 3 = Bladder, 4 = Rectum, 5 = Prostate). The dataset can be found [here](https://filesender.aarnet.edu.au/?s=download&token=76f406fd-f55d-497a-a2ae-48767c8acea2). See an example of an image and corresponding mask below:

![Example Images and Mask](./images/Image_and_Mask_Example.png)

### Preprocessing

- **Cropping:** Some images and masks are 256x128 while others are 256x144. To ensure uniformity, the 256x144 instances were cropped to 256x128. 
- **Normalisation:** Input images are normalised to the range [0, 1] by dividing by 255, then distributed to have zero mean and unit variance. This is important for stable learning. Importantly, masks are not normalised because their integer labelling needs to be preserved for loss and dice score calculations.
- **Train/Validation/Test Split:** The [dataset](https://filesender.aarnet.edu.au/?s=download&token=76f406fd-f55d-497a-a2ae-48767c8acea2) is pre-packaged into train/validation/test folders which is maintained for this project to ensure fairness in comparing results for similar projects working with this data. There are 11,460 instances in the training set, 660 instances in the validation set, and 540 instances in the testing set.
- **Loading NIfTI Images:** The NIfTI images are loaded into a Numpy array using the load_data_2D function in dataset.py. They are then converted and loaded into tensors of shape (B, 1, 256, 128) to use for the UNet, where B is the batch size which can be tuned.
- **Augmentation:** The Prostate and Rectum labels are fairly under-represented in the data. Trying to increase their representation, random rotations (up to 10 degrees) and random vertical flips were used as augmentations. Importantly, it is ensured that a transformation applied to an input image is identically applied to its corresponding mask to avoid misalignments.
- **Shuffling:** For the training set, the images are shuffled to ensure there is no learning of order that may be present. Shuffling need not be performed on the validation and testing sets.

## UNet Architecture
![UNet Architecture](./images/UNet%20Architecture.png)

### Overview
The UNet was initially developed for medical image segmentation. It uses a symmetric encoder-decoder architecture. The encoding path captures context while the decoding path enables precise localisation. The skip connections concatenate the encoder path and decoder path to recover spatial information lost during downsampling. All of these innovations improved on previous models which struggled with localisation.

### Components
**Encoder Path:**
   - **Double Convolution:** Each layer in the encoder path uses a double 3x3 convolution each followed by ReLU activations.
   - **Max Pooling:** Reduces the spatial dimensions, allowing the network to learn at multiple scales.
   - **Feature Doubling:** After the Max Pooling step, the next layer will use double the number of filters for its convolutions.
   - **Role:** The Double Convolutions are responsible for learning various features within the image of increasing complexity with each layer. The downsampling from Max Pooling allows the network to capture abstract information which makes the model invariant to small shifts and distortions. Feature Doubling allows the network to maintain its ability to represent features despite the reduction in spatial resolution.

**Bottleneck:**
   - **Double Convolution:** A double 3x3 convolution each followed by ReLU activations.
   - **Role:** Essentially part of the Encoder Path. It is the deepest layer in the UNet, bridging the Encoder and Decoder.

**Decoder Path:**
   - **Transposed Convolution:** A 2x2 Transposed Convolution to restore spatial dimensions.
   - **Skip Connections:** See below.
   - **Double Convolution:** Each layer in the encoder path uses a double 3x3 convolution each followed by ReLU activations.
   - **Role:** Recover spatial information via Skip Connections and restore spatial dimensions. The Double Convolutions learn about the combination of concatentated Decoder and Encoder features.

**Skip Connections:**
   - **Concatenation** The Double Convolution output from a layer in the Encoder Path is concatentated to the input for the Double Convolution in the corresponding layer of the Decoder Path.
   - **Role:** Restores spatial information lost during downsampling in the Encoder Path. Allows for more precise localisation of object boundaries. Merging feature maps from the Encoder Path with the Decoder Path allows the UNet to combine low-level details with high-level contextual information across the network.

**Output:**
   - **Final 1x1 Convolution:** Maps the final feature representation to an output of raw logits for the six labels in this task (convolution outputs channel dimension of size 6). Softmax can be applied to this output to get the predicted proabilities of each label for each pixel. Argmax can be applied to produce a visualisation for the predicted mask.

### References
Information about UNet architecture was retrieved from [here](https://viso.ai/deep-learning/u-net-a-comprehensive-guide-to-its-architecture-and-applications/). \
Figure was retrieved from [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

### Padding
Unlike in the original UNet architecture (see figure), zero-padding is used for the convolutions to preserve input and output image shapes.

## Model and Training Configuration

### Model
The UNet is setup with 4 layers. The number of filters doubles each layer: 64, 128, 256, 512.

### Loss Function
A custom CombinedLoss is defined, which is simply a weighted sum of Dice Loss and Cross-Entropy Loss. An even weight of 0.5 is used for both losses. This loss can also factor in label weights to assign more/less importance in learning specific labels in training. Equal label weights were used (1, 1, 1, 1, 1, 1), but for future work, this can potentially be tuned to give higher importance to the under-represented Rectum and Prostate labels for better performance.

To facilitate the calculation of Dice Score, the ground truth masks are one-hot encoded and the model's output has softmax applied to it to represent its predicted mask. A Dice Score is calculated for each label. The Dice Loss is then the average of 1 - Dice Scores. 


### Training Hyperparameters
Epochs: 50 \
Batch Size: 8 \
Learning Rate: 1e-4 \
Optimiser: Adam

### Training Process
The model takes batches of input images stored in tensors of shape (B, C, H, W) where B is the batch size, C is the number of channels for the input images (C = 1 for our greyscale images), H is the 256 pixel image height, and W is the 128 pixel image width. It then forwards this input through the network, yielding an output of shape (B, 6, 256, 128). The 6 channels represent the pixel-wise raw logits for the six labels. This output is then passed into the Loss Function along with the corresponding ground truth masks of shape (8, 1, 256, 128). The Loss Function calculates the Cross Entropy Loss and the Dice Loss, adding them together with equal 0.5 weighting to give the final loss. The model's parameters are then updated in a direction to reduce loss via gradient descent with the Adam optimiser. This process iterates through all the batches in the training set and repeats for a number of epochs with the aim of converging the model's parameters such that it can perform its image segmentation task well.

## Results

### Dice Scores
The trained UNet2D model achieved an average Dice Score of 0.985, 0.981, 0.922, 0.956, 0.758, and 0.702 on the test set for the Background, Body, Bone, Bladder, Rectum, and Prostate labels respectively (see predict.py output below). Only the Prostate label did not achieve the desired 0.75 minimum dice score, but came fairly close.

![Dice Scores](./images/Dice%20Scores.png)

### Visualisation of Sample Input Images, Ground Truth Masks, and the Model's Predicted Masks
In all the samples below, we can see that the model captures the background, the body, and the bladder quite well, but struggles with some of the finer details.

![Sample 0](./images/Sample%200.png)
![Sample 1](./images/Sample%201.png)
![Sample 2](./images/Sample%202.png)


### Loss Plot
![Loss Plot](./images/Loss%20Curves.png)

## Dependencies
Python: 3.12.4 \
PyTorch: 2.4.1 \
Numpy: 1.26.3 \
Matplotlib: 3.9.2 \
Nibabel: 5.3.1

## Directories
The directories to the data are defined in utils.py and can be changed as needed.

TRAIN_IMG_DIR: Path to directory containing all the training images \
TRAIN_MASK_DIR: Path to directory containing all the training masks \
VAL_IMG_DIR: Path to directory containing all the validation images \
VAL_MASK_DIR: Path to directory containing all the validation masks \
TEST_IMG_DIR: Path to directory containing all the testing images \
TEST_MASK_DIR: Path to directory containing all the testing masks

## Usage
Ensure data is downloaded from https://filesender.aarnet.edu.au/?s=download&token=76f406fd-f55d-497a-a2ae-48767c8acea2 \
Ensure an environment is setup with the listed dependencies installed \
Ensure the data paths are defined as required in utils.py \
To train the model: ```python train.py``` \
To get predictions after training the model: ```python predict.py``` \
Output visualisations are automatically saved to the working directory and Dice Score results are printed to stdout.

## Reproducibility
The code includes seed setting which allows the reported results to be reproduced.

## References
1. **UNet Original Paper:**
   * https://arxiv.org/abs/1505.04597
   * https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

2. **Dataset:**
   * https://filesender.aarnet.edu.au/?s=download&token=76f406fd-f55d-497a-a2ae-48767c8acea2

3. **UNet Architecture Diagram:**
   * https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

4. **Explanation of UNet**
   * https://viso.ai/deep-learning/u-net-a-comprehensive-guide-to-its-architecture-and-applications/