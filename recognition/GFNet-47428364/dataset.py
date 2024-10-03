import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class ADNIDataset(Dataset):
    def __init__(self, AD_dir, NC_dir):
        self.AD_dir = AD_dir
        self.NC_dir = NC_dir
        self.images = [(file, 1) for file in os.listdir(AD_dir)]
        self.images += [(file, 1) for file in os.listdir(NC_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.images[idx])
        image = Image.open(path).convert("RGB")
    
        if self.transform:
            image = self.transform(image)
        
        return image, self.label

# Data directory
dir = "recognition/GFNet-47428364/AD_NC"

# Initialise the datasets
train = ADNIDataset(dir + "/train/AD", dir + "/train/NC")
test = ADNIDataset(dir + "/test/AD", dir + "/test/NC")

# Creates that dataloaders
dataloader_train = DataLoader(train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test, batch_size=32, shuffle=True)