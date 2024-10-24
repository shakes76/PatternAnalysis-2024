# GFNet Vision Transformer for Alzheimer's Disease Classification

## Author
Chiao-Yu Wang (Student No. 48007506)

## Project Overview
Self-attention and pure multi-layer perceptrons (MLP) models have become increasingly popular for visual recognition tasks due to their ability to achieve high performance with fewer inductive biases. However, when high-resolution features are required, such models become harder to scale up since the complexity of self-attention and MLP grows significantly as image sizes increase. A new architecture, the Global Filter Network (GFNet), was therefore introduced in 2023 to address this limitation. This project aims to construct a GFNet which follows a similar structure to the original publication that can classify Alzheimer’s disease given a set of MRI brain scans and maintain a minimum accuracy of 80%.

## Global Filter Networks (GFNet)
The GFNet architecture is outlined in the publication ["GFNet: Global Filter Networks for Visual Recognition"](https://doi.org/10.1109/TPAMI.2023.3263824) and shares structural similarities with regular vision transformers. The main difference would be the replacement of the self-attention layer with a Global Filter Layer and a Feed Forward Network (FFN). The Global Filter Layers consists of a 2D discrete Fourier transform, an element-wise multiplication between frequency-domain features and learnable global filters, and a 2D inverse Fourier transform. The Feed Forward Network consists of a layer normalisation operation followed by a multilayer perceptron (MLP). The GFNet therefore contains the following layers: a patch embedding layer, Global Filter Layer, Feed Forward Network, Global Average Pooling layer and finally a Linear layer, which determines the class. The GFNet architecture is illustrated below.

<p align="center">
    <img src="images/gfnet.gif" alt="Global Filter Network Architecture">
</p>

## Project Dependencies
The dependencies listed below are recommended for replicating the results from this project.

- Python 3.11.5
- PyTorch 2.0.1
- TorchVision 0.15.2
- Matplotlib 3.8.4

## Repository Overview

`images/` contains the images used in this README.

`constants.py` contains the defined constants used for specifying data loading and model settings.

`modules.py` contains the components of the Vision Transformer.

`dataset.py` contains the function used for loading the data.

`train.py` contains the functions for compiling and training the model.

`predict.py` contains the functions for predicting on the trained model.

## Using the GFNet
### Parameters/Constants
Before training the model, set the global variables in `constants.py`. The variables are defined as follows:

`IMAGE_SIZE`: Height and width of each image.

`BATCH_SIZE`: Batch size of the training and testing data.

`LEARNING_RATE`: Learning rate of the Adam optimizer.

`NUM_EPOCHS`: Number of epochs to train the model for.

`WEIGHT_DECAY`: Weight decay of the Adam optimizer.

`TRAIN_DATA_PATH`: Path from which the training dataset will be loaded.

`TEST_DATA_PATH`: Path from which the test dataset will be loaded.

`MODEL_SAVE_PATH`: Path at which the trained model will be saved.

`DROPOUT_RATE`: Percentage of units to drop out in Multi-Layer Perceptron.

`SCHEDULER_FACTOR`: Factor by which the scheduler will reduce the learning rate.

`SCHEDULER_PATIENCE`: Number of epochs with no improvement after which the scheduler will reduce the learning rate.

### Building and Training the GFNet
- Running `train.py` will automatically train and build a GFNet model. The trained model will then be saved in the directory specified by `MODEL_SAVE_PATH`. Loss and accuracy curves for both training and validation will be plotted by Matplotlib.

- Running `predict.py` will predict results using the trained GFNet model. The trained model will be loaded from `MODEL_SAVE_PATH` and used to evaluate the test set. A confusion matrix will be plotted by Matplotlib and saved to the working directory.

## Dataset
The preprocessed ADNI brain dataset used within this project can be found [here](https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24). The original unprocessed dataset can also be found on the official [ADNI website](https://adni.loni.usc.edu/). The `TRAIN_DATA_PATH` and `TEST_DATA_PATH` variables in `constants.py` can be changed to specify another path from which the training and testing datasets should be loaded.

The dataset has been compiled for binary classification and as such, only consists of two classes: the `AD` class for brains with Alzheimer's Disease and the `NC` class for brains with Normal Cognitive function.

### Training, Validation and Test Splits
The preprocessed ADNI brain dataset consists of 21,520 images in the `train` folder and 9000 images in the `test` folder. Similar to Vision Transformers, GFNets require large amounts of training data to achieve high performance, so the validation set was derived from half of the `test` folder. This results in a dataset split of:
- 21,520 images in the training set
- 4500 images in the validation set
- 4500 images in the test set