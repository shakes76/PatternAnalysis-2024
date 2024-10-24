import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1" #Had a bug where albumentations module thought it had an update
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision

from modules import UNET
from dataset import ProstateCancerDataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    visualize_predictions,
    #save_predictions_as_img,
)


#HyperParameters
LEARN_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 6
NUM_WORKERS = 1
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False


TRAIN_IMG_DIR = 'C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_train'
TRAIN_SEG_DIR = 'C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_train'
VAL_IMG_DIR =  'C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_validate'
VAL_SEG_DIR = 'C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_validate'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        if targets.dim() == 4 and targets.shape[-1] == 5:  # Check if targets are one-hot encoded
            targets = torch.argmax(targets, dim=-1)
        #forward
        with torch.autocast(DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets.squeeze(1))
        
        #backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #Update tqdm loop
        loop.set_postfix(loss = loss.item())
            
    

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=256, width=128),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=256, width=128),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=1, out_channels=5).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_SEG_DIR,
        VAL_IMG_DIR,
        VAL_SEG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.amp.GradScaler(device = DEVICE)
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #check_accuracy(val_loader, model, DEVICE)
        if epoch == (NUM_EPOCHS-1):
            check_accuracy(val_loader, model, DEVICE)
            checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),}
            save_checkpoint(checkpoint)
            visualize_predictions(val_loader, model, device=DEVICE, num_images=3)

        #save model
        
        #print example
    

if __name__ == "__main__":
    main()