import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
import torchvision
from torchvision.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
    
    def load_data(self):
        images = []
        labels = []
        
        label_names = {"AD": 1, "NC": 0}
        sub_directory = "train" if self.train else "test"

        for label in label_names.keys():
            label_directory = os.path.join(self.root_dir, sub_directory, label)
            for img_name in os.listdir(label_directory):
                img_path = os.path.join(label_directory, img_name)
                images.append(img_path)
                labels.append(label_names[label])
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('L')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        
        return image, label