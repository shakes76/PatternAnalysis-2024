# Vision Transformers for the ADNI Dataset
This repository contains various implementations of modern vision transformers designed to be
applied to the Alzheimer's Disease Neuroimaging Initiative dataset (ADNI for short) [[1]](https://adni.loni.usc.edu/).
The dataset contains MRI scans of various subjects which have Alzheimer's (labeled as AD)
and those who are normal controls (labeled as NC). Each of the algorithms implemented aim to solve a classification
problem between these two classes, and are based of popular modern transformer architectures that have shown
high levels of performance in image based problems.

The standard vision transformer (ViT) model is based off the architecture proposed in the study  "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 
[[2]](https://arxiv.org/abs/2010.11929)

The convolutional vision transformer (CvT) is based off the architecture proposed in the study "CvT: Introducing Convolutions to Vision Transformers"
[[3]](https://arxiv.org/abs/2103.15808)

GFnet - To be implemented if time permits

Each of the non standard vision transformers build upon the original design, introducing new techniques to improve performance. The original 
ViT design stemmed from the standard transformer design, which has branched off to become integrated into may machine learning problems.
The ViT implemented one half of the original transformer architecture, the encoder, to solve imaged based problems. It offers improved
performance from CNN models which were dominating image based problems at the time the initial study was released. This improved performance 
is key to why the ViT model is such an effective solution to the classification problem of the ADNI dataset.

## Algorithms
### Vision Transformer
A visual representation of the ViT model architecture can be seen below.
![alt text](alzheimers_47446096/imgs/vit_architecture.jpg)
The ViT algorithm can be broken into some key steps
1. Patch projection using, traditionally using a linear layer. In the implementation of the ViT in this repository a convolutional layer
was used instead as this has bee shown to offer performance gains, this was noted in the paper as a 'Hybrid Architecture'. _Note: simply using a convolutional layer in this step of the ViT does not constitute a CvT as there fundamental differences between the two._
2. Positional encoding is then applied to the linear patch tokens to finalize the patch embedding process. This is done to give the model knowledge of the patches respective position in the original image. There are many methods available to achieve this, sinusoidal encoding was the chosen method for the implementation in this repository as it has shown to be extremely effective and is commonly used in ViT implementations.
Additionally, a learnable class token is applied, this is used to learn the information regarding classes and is extracted in the final
step to make the final classification.

3. The transformer encoder block, the embedded patches are then fed into the encoder block which performs two main actions, 
multi head self attention and standard MLP fully connected layer. This follows the standard transformer encoder architecture implemented in
numerous other models.

4. The final step is using the final values inside the learnable class token to make predictions, to do this these values are put into
a final MLP layer in which the final layer is used to make a classification.

### Convolutional Vision Transformer
A visual depiction of the CvT model architecture can be seen below.
![alt text](<alzheimers_47446096/imgs/CvT Architecture.png>)
The convolutional vision transformer builds upon the original ViT model by implementing a few major changes.
1. Firstly, the CvT model implements a multi stage downsizing process in which there is token embedding at
multiple points throughout the training process, aiming to progressively downsize the original image through each convolutional
embedding process. This allows for more detail to be retained through each stage when compared to the single embedding process in the ViT
model.
2. When calculating the multi head self attention layer, a forwards convolutional projection is used to calculate the 'query', 'key' and 'value'
values. In this implementation the stride of the convolutional layers of the key and value calculations have a stride of 2, this reduces the
size of the data giving performance gains, with negligible loss of detail as proven in the study proposing this architecture.
3. Another key difference between the CvT and ViT, is the removal of positional encodings, results from the study proposing this model showed
that removing positional encoding offered performance gains with negligible loss of accuracy.
4. The final major change to the ViT model in the CvT model is that the learnable class token is only added in the final stage, this reduces
the computation needed for the model to train while still preserving accuracy.
### GFNet
To be implemented if time permits
## Dependencies
<!-- TODO -->
<!-- TODO -->
TODO
<!-- TODO -->
<!-- TODO -->
## Usage / Examples
Below are details on how to use the implemented models to perform your own testing/train. The steps below replicate the process followed
to achieve the results shown in the Results section of the README, however this is not the only way the code in the repository can be used.

1. Begin by cloning the repository to your device and navigating into the 'alzheimers_47446096' folder
```
git clone https://github.com/Fuller81/PatternAnalysis-2024
cd .\recognition\alzheimers_47446096\
```
2. Then download the ADNI dataset and ensure the file structure is as follows. _Note, text with * next to it must match the values below exactly to ensure the relevant data preprocessing works as intended_
```
/ADNI Data set folder
    --/train *
        |--/AD
            |--/Patientxyz
            |--/Patientxyz
        |--/NC
            |--/Patientxyz
            |--/Patientxyz
    --/test *
        |--/AD
            |--/Patientxyz
            |--/Patientxyz
        |--/NC
            |--/Patientxyz
            |--/Patientxyz
/recognition *
    |--/alzheimers_47446096 *
```
### Data Pre Processing
#### 'Manual' Steps
Before training the data, two major manual preprocessing steps were taken. Creating a validation using a patient level split to ensure there
is no data leakage between the validation set and test set, and calculate the mean and standard deviation of the training data as this
is used in transforms applied to the data.
1. The first data preprocessing step was to create the validation set as follows. The functions in this repo designed to execute 
train/val/test split perform a 70/15/15 split. This is a common split ratio for smaller datasets. Due to the fact that the entire ADNI
dataset only contains around 30000 files it can be classified as a smaller dataset, and therefor it is suitable for this split.
    
    1.1 Ensure that the folder structure matches the pattern specified above

    1.2 Execute the 'formatByPatient' function which is located in the 'dataset.py' file. An example of this can be seen below.
    _Note, ROOT\_PATH and NEW\_ROOT\_PATH are relative paths from inside recognition\alzheimers\_47446096\\_
    ```
        > cd .\recognition\alzheimers_47446096\ 
        > python
        >>> from dataset import *
        >>> formatByPatient(path = ROOT_PATH, newPath = NEW_ROOT_PATH)
    ```
2. Calculate the mean and standard deviation of the training set.
    
    2.1 This is done by running the 'meanStdCalc' function which is also located in the 'dataset.py' file
    _Note, PATH is the relative path from inside recognition\alzheimers\_47446096\ to the train folder_
    ```
        > cd .\recognition\alzheimers_47446096\ 
        > python
        >>> from dataset import *
        >>> meanStdCalc(path = PATH)
    ```
    This will return the mean and standard deviation of each of the 3 channels in the original RGB images, however they should
    all be the same as the MRI photos are grey-scale already. To ensure that these values are being used for the normalisation
    transforms when the data is being loaded, change the global variables 'MEAN' and 'STD' in 'dataset.py'.

#### Transformations on Data Loading
When the data is loaded using the defined methods for getting dataLoaders for each dataset, several transforms are applied
in a effort to increase the performance of the models. The transforms applied are as follows

Transforms on training data (_All methods are pytorch transforms from torchvision.transforms_)
1. A random crop on the image of the predefined IMG_SIZE is performed using RandomResizedCrop((IMG_SIZE, IMG_SIZE)),
2. 3 random augmentations to the image using RandAugment() (A list of the included augmentations can be found [here](https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#RandAugment))
3. Convert the image into a single channel grey-scale format using Grayscale()
4. Convert the image into a torch tensor using ToTensor()
5. Normalize the images using the mean and standard deviation calculated above using Normalize(mean = MEAN, std = STD)

Transforms on test/val data (_All methods are pytorch transforms from torchvision.transforms_)
1. A center crop on the image of the predefined IMG_SIZE is performed using CenterCrop((IMG_SIZE, IMG_SIZE)),
3. Convert the image into a single channel grey-scale format using Grayscale()
4. Convert the image into a torch tensor using ToTensor()
5. Normalize the images using the mean and standard deviation calculated above using Normalize(mean = MEAN, std = STD)

These transform were chosen in an attempt to increase the performance of the model in various ways. The normalization, 
random crop and random augmentations were added in an attempt to create more variation in the dataset, thus giving 
the model 'more' samples to learn from, this gave a notable performance/accuracy increase compared to initial 
testing of the model without any transforms implemented. Additionally the decision to convert to grey-scale was made
for optimization purposes as since the image is inherently grey-scale and all RGB values are the same it allows for
reduced computation without loss of detail. 

### Training
Once the data processing steps listed above are completed, the model is then ready to be trained. To train the model first 
ensure that the correct model/hyperparameters are defined at the beginning of the 'train' function in 'train.py'. An exert from
'train.py' can be seen below highlighting where the model is defined. If desired other modifications to the optimiser, loss function,
mixed precision, etc. can also be changed within this file.
```python
def train(device: str = "cpu"):
    NUM_EPOCH = 1000
    LEARNING_RATE = 0.00003
    BATCH_SIZE = 64
    WEIGHT_DECAY = 0.02
    TRAIN_WORKERS = 4

    #* Define Model Here ------------------------------------
    model = ConvolutionalVisionTransformer(device).to(device)
```
The file can then by run to begin training the model. _Note ensure you have navigated to the same directory where you relative paths for 'dataset.py' are based from. Otherwise getting the dataLoaders will error._

```
> cd .\recognition\alzheimers_47446096\ 
> cd python .\train.py
```
An example of the first few lines of output can be seen below
```
Beginning Training
Epoch 1/1000: Training Loss = 0.6941055584732079, Training Accuracy = 0.5199349442379182
Validation Loss = 0.6975890823772976, Validation Accuracy = 50.73660714285714 %
Epoch 2/1000: Training Loss = 0.6923188280634781, Training Accuracy = 0.5276486988847584
Validation Loss = 0.6800375240189689, Validation Accuracy = 60.04464285714286 %
```
### Testing / Predicting
To test a saved model

```
Test Accuracy = 73.49557522123894 %
F1 Score: 0.7711993888464477
Confusion Matrix:
 [[1303  937]
 [ 261 2019]]
```

## Results
When testing with both the 
<p align="center">
  <img src="alzheimers_47446096/model15/trainAccuracyPlot.jpg" />
</p>
<p align="center">
  <img src="alzheimers_47446096/model15/trainLossPlot.jpg" />
</p>
<p align="center">
  <img src="alzheimers_47446096/model15/valAccuracyPlot.jpg" />
</p>
<p align="center">
  <img src="alzheimers_47446096/model15/valLossPlot.jpg" />
</p>

## References
[[1]](https://adni.loni.usc.edu/)
[[2]](https://arxiv.org/abs/2010.11929)
[[3]](https://arxiv.org/abs/2103.15808)