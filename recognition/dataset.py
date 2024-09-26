import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data_2D, to_channels
import nibabel as nib
import os, glob

"""
The files are stored in .NII format.
"""

trainPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
testPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

tempTrain = []
tempTest = []

for file in os.listdir(trainPath):
    tempTrain.append(os.path.join(trainPath, file))

for file in os.listdir(testPath):
    tempTest.append(os.path.join(testPath, file))

print("Getting training data...")
trainSet = load_data_2D(tempTrain)
print(trainSet[0].shape)
print("Getting testing data...")
testSet = load_data_2D(tempTest)
print(testSet[0].shape)
print("Finished fetching data.")




    
    


