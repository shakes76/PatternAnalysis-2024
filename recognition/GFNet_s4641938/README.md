# Implementation of Computer Vision Neural Network for ADNI Dataset (Problem 5)

## Introduction

This repository contains code to train a computer vision neural network designed to classify images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to assist in the classification and understanding of Alzheimer's disease progression through deep learning techniques. This repository folder contains an implementation of [GFNet](https://ieeexplore.ieee.org/document/10091201).

## About the Model

[GFNet](https://ieeexplore.ieee.org/document/10091201) is a cutting-edge vision transformer neural network that prizes itself on efficiently capturing spatial interactions through its use of the fast fourier transform.

![GFNet Structure](https://github.com/user-attachments/assets/b8e67323-a4d2-4427-ac7c-0e3720ccc62a)

[The above image was created by the GFNet Authors](https://github.com/raoyongming/GFNet)

GFNet adapts the well-known vision transformer (ViT) models by replacing the self-attention layer with a global filter layer. 
GFNet contains:
- **Patch Embedding**: Initial input images are split into several smaller size patches, which are then flattened into a lower-dimensional space.
- **Global Filter Layer**: The fast fourier transform is used to find the spatial interactions between the data.
- **Feed Forward Network**: A Multi-Layer Perceptron processes the results from the global filter layer through a activation function to  learn non-linear transformations, improving the model's ability to learn features
- **Global Average Pooling and Classification**: After the N sections of Global Filter Layer & Feed Forward Network, the resulting information is pooled together then used to classify.
- **Normalisation layers**: Optional normalisation layers to normalise the values between each section, helping improve generalisability.


### Model Architecture
- **Base Model**: GFNet
- **Input Shape**: 180x180
- **Output Classes**: Alzheimer's (AD), Normal Cognition (NC)
- **Framework**: PyTorch

# How to use

## Training
Parameters
```zsh
python ./recognition/GFNet_s4641938/train.py [IMAGESIZE] [EPOCHS] [ADNIROOTDATAPATH]
```
Where:
- **IMAGESIZE**:contains the image sizes after preprocessing (e.g. 180 means 180x180)
- **EPOCHS**: contains number of epochs to train for
- **ADNIROOTDATAPATH**: contains the path to the ADNI database where the file passed contains files in the format:

```
│ADNI/
├──train/
│  ├── AD
│  │   ├── 369883_4.jpeg
│  │   ├── 369883_5.jpeg
│  │   ├── ......
│  ├── NC
│  │   ├── 370202_42.jpeg
│  │   ├── 370202_43.jpeg
│  │   ├── ......
├──test/
│  ├── AD
│  │   ├── 370001_94.jpeg
│  │   ├── 370001_95.jpeg
│  │   ├── ......
│  ├── NC
│  │   ├── 370323_53.jpeg
│  │   ├── 370323_54.jpeg
│  │   ├── ......
```

Example
```zsh
python ./recognition/GFNet_s4641938/train.py 180 50 "/home/groups/comp3710/ADNI/AD_NC"
```

During training, the ongoing best model will be saved at ./best_model.pth

## Testing
After training
Parameters
```zsh
python ./recognition/GFNet_s4641938/predict.py [IMAGESIZE] [MODELPATH] [ADNIROOTDATAPATH]
```

Where:
- **IMAGESIZE**:contains the image sizes after preprocessing (e.g. 180 means 180x180)
- **MODELPATH**: contains the path to the model to assess
- **ADNIROOTDATAPATH**: contains the path to the ADNI database where the file passed contains files in the format:

Example
```zsh
python recognition/GFNet_s4641938/predict.py 180 "./best_model.pth" "/home/groups/comp3710/ADNI/AD_NC"
```

## Training Details

### Dataset
- **Source**: Alzheimer's Disease Neuroimaging Initiative (ADNI)
- **Training** 25120 (256x240) images
- **Test** 9000 (256x240) images
- **Preprocessing**: Images were resized, converted to grayscale and normalized to improve the training quality from the data.
- **Train/Validation/Test Split** 25120/0/9000. The best performing model uses 25120/0/9000 training/validation/test split since due to the limited size of the dataset it would be better to use all data provided, but the training dataset was split 90%/10% training/validation in later versions (see challenges for steps and changes made).

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Epochs**: 50
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Loss Function**: Cross-Entropy Loss

### GFNet Model Parameters:
- **Patch Size**: 16
- **Embed Dimensions**: 512
- **Depth**: 18
- **MLP ratio**: 4
- **Drop path rate**: 0.25

### Training Procedure
1. **Load the Dataset**: Use `torchvision.datasets` and `torch.utils.data.DataLoader` to load and preprocess the ADNI dataset.
2. **Define the Model**: Instantiate the GFNet model following the given model parameters.
3. **Train the Model**: Execute training loops, monitor accuracy and loss, saving the best performing model after each epoch.
4. **Assess Performance**: After training has concluded, use predict.py to show the performance of the best-performing model.

# Dependencies
- **Python**
- **PyTorch**
- **torchvision**
- **timm** >= 1.8.0

# Performance & Results
The model achieved the following results on the validation set:
- **Test Accuracy**: 62.9%
- **Learning Rate**: 0.0001
- **Epochs**: 50

## Loss/Accuracy Plot
![accuracyPlot](https://github.com/user-attachments/assets/ef0e3191-245a-4026-a393-0347dc81562c)

Here it is evident that the current model suffers from a significant amount of overfitting. 
In the report documentation below I discuss the methods I used to attempt to improve/fix this.
Inherently, this would be a large issue given the size of the dataset. ViT models are built to derive
spatial and image information from extremally large datasets (often 1+ million), while the training data
available only contains around ~25000 images. 

# Best Model Location
The link below provides the file of the model that performs to the accuracy described, since the training may vary between each training loop due to random choices made by pytorch/numpy during training:

[Google Drive Download Link](https://drive.google.com/file/d/1GHfTqKdfuTHTDx1CqEyr2Qh0SxeQvd6Y/view?usp=sharing)

Due to the file size of the model (~150MB), it is available on the google drive link above.


# Report & Process Documentation
###
Given the scale of the problem, while it is possible to add several sets of GFNet blocks together, the first choice was to try 1 GFNet block given the scale of the dataset and problem. Some of the important initial parameters used were:
- **Learning Rate**: 0.01
- **Epochs**: 100
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

A learning rate of 0.01 caused the loss to significantl oscillate during training, and struggled to converge, so the learning rate was reduced to 0.001, then 0.00001.

100 epochs resulted in a model that was capable of perfect accuracy on the training set, but only 54% on the test set, which resulting in a far too overfitted model - therefore the epochs were reduced and models were saved based on the best accuracy at the time.

When the overfitting was still a notable issue, Adam was changed to AdamW to attempt to help the model generalise by making it choose the parameters that generalised the model best.

### Challenges
The main challenge with this model has been overfitting.
Overfitting can easily be seen on the plot above:
(training accuracy goes to ~1.0, while test accuracy stagnates around ~0.62).

Various methods into data pre-processing and model scaling were investigated:
- **Pre-processing**: Modifying the training data by adding Gaussian noise, Random Erasure, Color Jitter
- **Model**: Mixture of complex multi-layer GFNet Pyramid models to simplistic single-layer GFNets
- **Dropout**: Adding dropout and normalisation layers and tested various levels

More complex models did no help convergence, since the issue was overfitting.

Throughout the process I added various different levels of dropout: In GFNet there are 2 values for GFNet (drop_rate and drop_path_rate).
'drop_rate' refers to a flat dropout rate in each global filter layer, while drop_path_rate refers to a linearly increasing drop rate. 
Updating the drop_rate or drop_path rate did not increase the overfitting issue.
There were additional transformations made to the images: random erasure, color jitter, gaussian noise.
None of the changes made any notable improvement to the test accuracy. 

Unfortunately, this leads into a major issue of a vision transformer mentioned before. 
It expects a significant amount of divergent data to train on. The original version of GFNet as originally implemented was built to handle
a subset of imageNet, which contains 14 million labelled images in total. The subset used for the model contains 1000 categories, 100000 training images and 50000 test images. The image information GFNet is expected to learn has significantly more variety in image appearence (dog, cat, house, tree,...) than for the ADNI dataset, containing 25000 MRI brain scan slices. 

Unfortunately none of the attempted changes were able to bring improvements to the overall test performance.

To test if overfitting could be stopped, I shrunk the size of the GFNet to tiny sizes with the following parameters:

- **Learning rate**: 0.005
- **Epochs**: 100
- **Patch Size**: 4 (25% of Patch Size in best model)
- **Embed Dimensions**: 4 (~1% of Embed Dim in best model)
- **Depth**: 2 (~11.1% of Depth in best model)
- **MLP ratio**: 1 (25% of MLP ratio in best model)
- **Drop path rate**: 0.1

This model produced the following results:

![TinyGFNet](https://github.com/user-attachments/assets/df78bbef-4053-47ac-b825-0bec1219ced9)

It takes notably longer for the training accuracy to increase, but the test accuracy does not improve from decreasing the model.
Therefore overall there is a minor reduction in overfitting, but the test accuracy does not improve.

To help benefit any further research or investigation into this model, any further work on this model's performance would benefit with this in mind. 

Another potential route I began to work on is the idea of breaking up the batches into patients, since that would potentially help GFNet to understand that these images come from the same patient. This was implemented and run on rangpur, but the performance failed to improve above 60%. The training data for patients was split 90%/10% train/validation to better assess model performance:

![Large Acc](https://github.com/user-attachments/assets/b52f27a5-266d-432b-9ff7-07eb506934b8)

This was done by preprocessing the data such that each batch called by the dataloader would give back all the images for the number of patients. The data was then flattened into a 4D vector that corresponded to (images, channels, height, width) from all patient images. For example, a batch size of 3 would get the images for 3 random patients, then the model would train on all the images for those patients at once for that batch.

On a more detailed investigation, I found that the original weight initalisation caused all predictions to initalise as negative. (So 100% guessed normal cognition) (see example result on test data in below table:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | 0  | 4460 |
| **Actual Negative** | 0 | 4540  |

The model have improved given enough training time, but this was not feasible due to time constraints. Regardless, I attempted to change the initalisation of the parameters to have significant more deviation in the weight parameters:
Original (see function in modules.py inside the GFNetPyramid class)
```python
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
```
Increased deviation:
```python
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.15) #Increased variance of weights in all linear layers from 0.02 -> 0.15
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
```

This produced the following confusion matrices:
After training for 1 epoch:
|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | 1553  | 1417 |
| **Actual Negative** | 2907 | 3123  |

Accuracy: 0.520

After training for 400 epochs:
|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | 1564  | 1391 |
| **Actual Negative** | 2896 | 3149  |

Accuracy: 0.524


Therefore, the change to weight initalisation gave a better distribution to predictions, but unfortunately training for 400 epochs failed to not produce a noticable improvement in test accuracy.

Due to time constraints, I was unable to further test additional methods, like longer learning times or other preprocessing or changing the model parameters, etc further. 

## License
MIT License

### Acknowledgments
Many thanks to the Alzheimer's Disease Neuroimaging Initiative for creating the dataset used in this project.

An additional thanks to [Chandar Shakes](https://github.com/shakes76) to providing the cleaned ADNI brain dataset used in training this model. 

The file modules.py contains the GFNet implemented as desribed in the paper [Global Filter Networks for Image Classification](https://arxiv.org/abs/2107.00645) by Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou ([GitHub](https://github.com/raoyongming/GFNet))
