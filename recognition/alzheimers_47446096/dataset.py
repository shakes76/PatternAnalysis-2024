import torch
import os
import pandas as pd
from torch.utils.data import Dataset

TRAINPATH = "../ADNI/AD_NC/train"

def makeAnotateFile(path: str) -> str:
    '''
    Creates file with image label/class combinations
    Input: path - path to folder that contains images seperated by class into
        folders as seen below
        /path
            /class1
                img1
                img2
            /class2
            /class3
            /etc
    Returns: csv style string with image label combinations
    '''
    data = pd.DataFrame(columns = ["Label", "Class"])
    classVal = 0
    for classFolder in os.listdir(path):
        for imgLabel in os.listdir(os.path.join(path, classFolder)):
            data.loc[len(data)] = [imgLabel, classVal]
        classVal += 1
    return data.to_csv(index = False, header = False)