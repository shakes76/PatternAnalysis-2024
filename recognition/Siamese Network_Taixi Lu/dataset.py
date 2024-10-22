import torch
import random
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class ISICPairDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0] + '.jpg')
        image1 = Image.open(img1_name).convert("RGB")
        label1 = self.annotations.iloc[idx, 1]

        # Randomly select another image to form a pair
        pair_idx = random.randint(0, len(self.annotations) - 1)
        img2_name = os.path.join(self.img_dir, self.annotations.iloc[pair_idx, 0] + '.jpg')
        image2 = Image.open(img2_name).convert("RGB")
        label2 = self.annotations.iloc[pair_idx, 1]

        # Create label for the pair (1 if same class, 0 if different)
        label = 1 if label1 == label2 else 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label


def get_pair_data_loader(csv_file, img_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ISICPairDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader
