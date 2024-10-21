import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ADNI_Dataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        """
            image_dir (str): the path that store the .jpeg picture.
            labels (list): to distinguish between NC and AD, NC is 0, AD is 1.
            transform (callable, optional): the optional for dataset, the default is None.
        """
        self.image_dir = image_dir
        self.labels = labels
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        # return the size of data
        return len(self.images)

    def __getitem__(self, idx):
        # load the picture and label by idx.
        img_name = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_name).convert('L')  # change jpeg to greyscale.

        # If has optional, use it.
        if self.transform:
            img = self.transform(img)

        # get the label
        label = self.labels[idx]

        return img, label

def get_data_loaders(image_dir, labels, batch_size=8, shuffle=True):
    # define the transform options
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # To model size.
        transforms.ToTensor(),  # To tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ADNI_Dataset(image_dir, labels, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
