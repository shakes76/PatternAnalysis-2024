"""
Load and preprocess data

"""
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ADNIDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # get the path of AD and NC directory
        self.ad_path = os.path.join(folder_path, "AD")
        self.nc_path = os.path.join(folder_path, "NC")

        # Create lists for image path and corresponding label
        self.image_paths = []
        self.labels = []

        for label, class_dir in enumerate([self.ad_path, self.nc_path]):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".jpeg"):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label) # set class AD as 0, class NC as 1


    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create train and test dataset
train_dataset = ADNIDataset(folder_path="./ADNI/AD_NC/train", transform=transform)
test_dataset =ADNIDataset(folder_path="./ADNI/AD_NC/test", transform=transform)

# Load train and test set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# # check
# def check_dataloader(loader, name):
    
#     data_iter = iter(loader)
#     images, labels = next(data_iter)

#     print(f"\ncheck {name} dataloader:")
#     print(f"batch image size: {images.shape}")  # should be [batch_size, 3, 224, 224]
#     print(f"batch labels: {labels}")  # len should batch_size tensor

#     # ensure image tensor in a range of [0, 1] if necessary
#     print(f"image tensor min: {torch.min(images)}")
#     print(f"image tensor max: {torch.max(images)}")
#     print(f"image tensor avg: {torch.mean(images)}")

# # chack training dataloader
# check_dataloader(train_loader, "training set")

# # check test dataloader
# check_dataloader(test_loader, "test set")
