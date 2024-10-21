from utils import load_data_2D
import os

"""
The files are stored in .NII format.
"""

trainPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
testPath = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

def GetTrainSet():
    tempTrain = []

    for file in os.listdir(trainPath):
        tempTrain.append(os.path.join(trainPath, file))

    print("Getting training data...")
    trainSet = load_data_2D(tempTrain, normImage=True)
    print(trainSet[0].shape)

    print("Finished fetching training data.")

    return trainSet

def GetTestSet():
    tempTest = []

    for file in os.listdir(testPath):
        tempTest.append(os.path.join(testPath, file))

    print("Getting testing data...")
    testSet = load_data_2D(tempTest, normImage=True)
    print(testSet[0].shape)

    print("Finished fetching testing data.")

    return testSet

    


