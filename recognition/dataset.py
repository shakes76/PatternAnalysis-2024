# Import necessary libraries and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from glob import glob
from sklearn.model_selection import train_test_split
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 8
NUMBER_EPOCHS = 100
IMG_SIZE = 256

def imshow(img, title=None):
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get paths of images and annotations
train_input_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
train_gt_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"

train_image_paths = sorted(glob(os.path.join(train_input_dir, "*.jpg")))
train_mask_paths = sorted(glob(os.path.join(train_gt_dir, "*.png")))

# Create mapping between images and annotations
image_dict = {os.path.basename(x).split('.')[0]: x for x in train_image_paths}

mask_dict = {}
for mask_path in train_mask_paths:
    mask_name = os.path.basename(mask_path)
    image_name = mask_name.replace('_segmentation', '').split('.')[0]
    mask_dict[image_name] = mask_path

data_pairs = []
for image_name, image_path in image_dict.items():
    if image_name in mask_dict:
        mask_path = mask_dict[image_name]
        data_pairs.append((image_path, mask_path))
    else:
        print(f"Mask not found for image {image_name}")

# Split into training and validation sets
train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Define dataset class
class ISICSegmentationDataset(Dataset):
    def __init__(self, image_mask_pairs, transform=None):
        self.image_mask_pairs = image_mask_pairs
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, mask_path = self.image_mask_pairs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.image_mask_pairs)

# Define data transformations

class SegmentationTransform:
    def __init__(self, img_size):
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
    def __call__(self, img, mask):
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = mask.squeeze(0)
        mask = (mask > 0.5).long()
        return img, mask

# Create datasets and data loaders
transform = SegmentationTransform(IMG_SIZE)
trainset = ISICSegmentationDataset(train_pairs, transform=transform)
valset = ISICSegmentationDataset(val_pairs, transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# Visualize some images and annotations
import torchvision
dataiter = iter(trainloader)
images, masks = next(dataiter)
imshow(torchvision.utils.make_grid(images), title='Images')

def show_masks(masks):
    masks = masks.numpy()
    fig, axs = plt.subplots(1, masks.shape[0], figsize=(15, 5))
    for i in range(masks.shape[0]):
        axs[i].imshow(masks[i], cmap='gray')
        axs[i].axis('off')
    plt.show()

show_masks(masks)

# Define model and training process
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.encoder = models.vgg16_bn(pretrained=True).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = UNet(n_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(NUMBER_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in trainloader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{NUMBER_EPOCHS}], Loss: {epoch_loss:.4f}")
    
      # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in valloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(valloader)
    print(f"Validation Loss: {val_loss:.4f}")
