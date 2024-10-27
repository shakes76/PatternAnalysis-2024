import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

import time
import math
import matplotlib.pyplot as plt
import os



# create hyper paramaters
new_size = 128

# create dataset class to store all the images
class ISIC2018DataSet(Dataset):
    def __init__(self, imgs_path, labels_path, transform=None, labelTransform=None):
        self.LabelsPath = labels_path
        self.ImagesPath = imgs_path
        self.LabelNames = os.listdir(self.LabelsPath)
        self.imageNames = os.listdir(self.ImagesPath)
        self.LabelsSize = len(self.LabelNames)
        self.ImagesSize = len(self.imageNames)
        self.transform = transform
        self.labelTransform = labelTransform

    def __len__(self):
        if self.ImagesSize != self.LabelsSize:
            print("Bad Data! Please Check Data, or unpredictable behaviour!")
            return -1
        else:
            return self.ImagesSize

    def __getitem__(self, idx):
        img_path = os.path.join(self.ImagesPath, self.imageNames[idx])

        # This accounts for an invisible .db file in the test folder that can't be removed
        if img_path == "isic_data/ISIC2018_Task1-2_Test_Input\Thumbs.db":
            img_path = "isic_data/ISIC2018_Task1-2_Test_Input\ISIC_0036309.jpg"
            self.imageNames[idx] = "ISIC_0036309.jpg"
        image = read_image(img_path)
        label_path = os.path.join(self.LabelsPath, self.imageNames[idx].removesuffix(".jpg") + "_segmentation.png")
        label = read_image(label_path)

        if self.transform:
            image = self.transform(image)
        if self.labelTransform:
            label = self.labelTransform(label)
        
        return image, label




# functions to transform the images
def img_transform():

    img_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((new_size, new_size))
    ])

    return img_tr

def label_transform():
        
    label_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((new_size, new_size))
    ])

    return label_tr
