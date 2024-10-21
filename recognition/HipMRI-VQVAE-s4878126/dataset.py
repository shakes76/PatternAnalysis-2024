from utils import load_data_2D
import os

"""
The files are stored in .NII format.
"""

trainPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
testPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
valPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"


"""
Get the training set from Rangpur.
The training set contains over 11,000 MRI Scans.
"""
def GetTrainSet():
    tempTrain = []

    for file in os.listdir(trainPath):
        tempTrain.append(os.path.join(trainPath, file))

    print("Getting training data...")
    trainSet = load_data_2D(tempTrain, normImage=True)
    print(trainSet[0].shape)

    print("Finished fetching training data.")

    return trainSet

"""
Get the testing set from Rangpur. 
The testing set contains 540 MRI Scans.
"""
def GetTestSet():
    tempTest = []

    for file in os.listdir(testPath):
        tempTest.append(os.path.join(testPath, file))

    print("Getting testing data...")
    testSet = load_data_2D(tempTest, normImage=True)
    print(testSet[0].shape)

    print("Finished fetching testing data.")

    return testSet

"""
Get the validation set from Rangpur.
"""
def GetValSet():
    tempVal = []

    for file in os.listdir(valPath):
        tempVal.append(os.path.join(valPath, file))

    print("Getting testing data...")
    valSet = load_data_2D(tempVal, normImage=True)
    print(valSet[0].shape)

    print("Finished fetching testing data.")

    return valSet

    


