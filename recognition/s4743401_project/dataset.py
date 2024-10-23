import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
data_dir = '/Users/gghollyd/comp3710/report/AD_NC/'
def dataloader(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #check normalisation and where done. See if better here or in predict idk.
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
