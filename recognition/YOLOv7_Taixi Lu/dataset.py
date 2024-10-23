import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

g_image_size = (640, 640)

# Data preprocessing transformation
image_transform = transforms.Compose([
    transforms.Resize(g_image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ISICDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, "ISIC-2017_Training_Data", img_name)
        if not img_path.endswith('.jpg'):
            img_path += ".jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.annotations.iloc[idx][['melanoma', 'seborrheic_keratosis']].astype(int).values
        if label[0] == 1:
            label = 1
        elif label[1] == 1:
            label = 2
        else:
            label = 0
        # I have checked the dataset, No data is melanoma and seborrheic_keratosis the same time
        return image, label, img_name

# Data loader function
def get_dataloader(csv_file, root_dir, batch_size=8, shuffle=True):
    dataset = ISICDataset(csv_file, root_dir, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader