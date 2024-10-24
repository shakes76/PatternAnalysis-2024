import os
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
    #save_predictions_as_img,
)

#image_folder = 'keras_slices_train'
#imageNames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii.gz')]

#images= load_data_2D(imageNames)

# Check the shape and content
#print(f"Number of images loaded: {len(imageNames)}")
#print(f"Shape of first image: {images[0].shape}")

#HyperParameters
LEARN_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
'''
TRAIN_IMG_DIR = 'recognition/46976916-HipMRI/keras_slices_train'
TRAIN_SEG_DIR = 'recognition/46976916-HipMRI/keras_slices_seg_train'
VAL_IMG_DIR =  'recognition/46976916-HipMRI/keras_slices_validate'
VAL_SEG_DIR = 'recognition/46976916-HipMRI/keras_slices_seg_validate'
'''

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
            #print(f"Predictions shape: {predictions.shape}")
            #print(f"Targets shape before squeezing: {targets.shape}")
            loss = loss_fn(predictions, targets.squeeze(1))
        
        #backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #Update tqdm loop
        loop.set_postfix(loss = loss.item())
            
    

def main():
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(f"Using device: {DEVICE}")
    '''
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
            ToTensorV2,
        ],
    )

    val_transforms = A.compose(
        [
            A.Resize(height=256, width=128),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2,
        ],
    )
    '''
    train_transform = A.Compose([
        ToTensorV2(),  # Only convert to tensor without any other transformations
    ])

    val_transforms = A.Compose([
        ToTensorV2(),  # Only convert to tensor without any other transformations
    ])


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
        #print("started an epoch")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #print("completed an epoch")

        #save model
        #check accuracy
        #print example
    

    #dataset = ProstateCancerDataset(TRAIN_IMG_DIR, TRAIN_SEG_DIR)

    #image, segImage = dataset[0]

    #print("Images type:", type(dataset))
    #print("Image type:", type(image))
    #print("SegImages type:", type(segImage))

    #print("Image shape:", image.shape)  # Should print something like (1, H, W) where H, W are image dimensions
    #print("Ground truth shape:", segImage.shape)
    #plt.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale display
    #plt.title(f'Image 0')
    #plt.axis('off')  # Turn off axis labels
    #plt.show()

if __name__ == "__main__":
    main()