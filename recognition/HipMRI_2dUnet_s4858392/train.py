import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import uNet
from newdataset import NIFTIDataset
import  torch.nn.functional as F

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

if __name__ == '__main__':
    #images dataset
    train_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
    val_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"

    #mask dataset
    train_seg_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
    val_seg_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

    train_dataset = NIFTIDataset(imageDir=train_image_dir)
    val_dataset = NIFTIDataset(imageDir=val_image_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = uNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    def validate(model, loader, criterion, device):
        model.eval()
        val_loss = 0.0
        total_dice_score = 0.0
        with torch.no_grad():
            for batch  in loader:

                images, masks = batch
                images = images.to(device)
                masks =masks.to(device)
                if images.dim() ==3:
                    images = images.unsqueeze(1)
                if masks.dim() ==3:
                    masks = masks.unsqueeze(1)
                outputs = model(images)
                masks_resized = F.interpolate(masks, size=(256, 128), mode="nearest")
                loss = criterion(outputs, masks_resized)
                val_loss += loss.item()

        avg_loss = val_loss / len(loader)
        return avg_loss

    def train(model, loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        for images, masks in loader:
            if len(images.shape) ==3:
                images = images.unsqueeze(1)
            images, masks = images.to(device), masks.to(device)
            if masks.dim()==3:
                masks = masks.unsqueeze(1)
            # Zero the gradients
            optimizer.zero_grad()
            outputs = model(images)
            # Forward pass
            if outputs.dim() ==4:
                outputs = outputs.view(outputs.size(0),-1,outputs.size(2),outputs.size(3))
            masks_resized = F.interpolate(masks, size=(256, 128), mode="nearest")
            loss = criterion(outputs, masks_resized)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        return avg_loss
#Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")