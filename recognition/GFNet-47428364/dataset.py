import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ADNIDataset(Dataset):
    def __init__(self, AD_dir, NC_dir, transform=None):
        self.AD_dir = AD_dir
        self.NC_dir = NC_dir
        self.transform = transform
        self.images = [(os.path.join(AD_dir, file), 1) for file in os.listdir(AD_dir)]
        self.images += [(os.path.join(NC_dir, file), 0) for file in os.listdir(NC_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert("RGB")
    
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data directory
dir = "recognition/GFNet-47428364/AD_NC"

# Transform the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Initialise the datasets
train = ADNIDataset(dir + "/train/AD", dir + "/train/NC", transform=transform)
test = ADNIDataset(dir + "/test/AD", dir + "/test/NC", transform=transform)

# Creates that dataloaders
dataloader_train = DataLoader(train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test, batch_size=32, shuffle=True)