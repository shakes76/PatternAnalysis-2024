"""
containing the data loader for loading and preprocessing your data
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): The root directory of the dataset
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        """
        self.root = root 
        self.transform = transform

        self.files = sorted(os.listdir(root))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.files[index]))

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)



# Function to get the dataloader
def get_dataloader(root, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageDataset(root, transform)

    return DataLoader(dataset, batch_size=batch_size)