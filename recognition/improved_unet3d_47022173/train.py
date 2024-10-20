"""
This file contains the code to train the 3D U-Net model on the training set and validate on the 
validation set. The training loops contains validation and saving of the model every few epochs.
A final trained version of the model is saved at the end of training.
"""

from dataset import *
import torch
from torch.utils.data import DataLoader
import torchio as tio
from torch.nn import functional as F
from utils import *
from modules import *
from monai.losses.dice import DiceLoss
import nibabel as nib
from matplotlib import pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

if IS_RANGPUR:
    images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/"
    masks_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/"
    epochs = 50
    batch_size = 5
else:
    images_path = "./data/semantic_MRs_anon/"
    masks_path = "./data/semantic_labels_anon/"
    epochs = 10
    batch_size = 2



# Data parameters
batch_size = batch_size
shuffle = True
num_workers = 2
epochs = epochs
background = False

# Model parameters
in_channels = 1 # greyscale
n_classes = 6 # 6 different values in mask
base_n_filter = 8

# Optimizer parameters
lr = 1e-3
weight_decay = 1e-2

# Scheduler parameters
step_size = 10
gamma = 0.1


def save(predictions, affines, epoch): 
    predictions = torch.argmax(predictions, dim=1)
    batch = int(predictions.shape[0] / 1 / 128 / 128 / 128)
    predictions = predictions.view(batch, 1, 128, 128, 128).cpu().numpy()
    prediction = predictions[0].squeeze() # Take first, remove batch dimension
    affine = affines.numpy()[0] # Take first
    nib.save(nib.Nifti1Image(prediction, affine, dtype=np.dtype('int64')), f"saves/prediction_{epoch}.nii.gz")

# def dice_score_per_channel(preds, masks, smooth=1e-6):  
#     dice_scores = []
#     for c in range(n_classes):
#         pred = preds[:, c, ...]
#         target = masks[:, c, ...]
        
#         intersection = (pred * target).sum(dim=(1, 2, 3))  # Sum over 3D spatial dimensions
#         pred_sum = pred.sum(dim=(1, 2, 3))
#         target_sum = target.sum(dim=(1, 2, 3))
        
#         dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
#         dice_scores.append(dice.mean())  # Take mean over batch dimension
    
#     return dice_scores

def validate(model, data_loader):
    model.eval()  
    dice_score = DiceLoss(softmax=True, include_background=background)
    with torch.no_grad():
        dice_scores = [0] * n_classes
        for i, data in enumerate(data_loader):  
            inputs, masks, affine = data
            inputs, masks = inputs.to(device), masks.to(device)
            one_hot_masks_3d = F.one_hot(masks, num_classes=6).permute(0, 4, 1, 2, 3)

            # Forward pass
            softmax_logits, predictions, logits = model(inputs)
            
            for i in range(n_classes):
                logits = logits[:, i, ...]
                masks = one_hot_masks_3d[:, i, ...]
                dice_scores[i] += dice_score(logits, masks)
                
    average_dice_scores = [score / len(data_loader) for score in dice_scores]
    print(f"Average Dice Score: {list(map(lambda x: float(x / len(data_loader)), average_dice_scores))}")


if __name__ == '__main__':
    # Load and process data
    train_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.RandomFlip(),
        tio.Resize((128,128,128)),
        tio.RandomAffine(degrees=10),
        tio.RandomElasticDeformation(),
        tio.ZNormalization(),
    ])

    valid_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.Resize((128,128,128)),
        tio.ZNormalization(),
    ])
    
    valid_dataset = ProstateDataset3D(images_path, masks_path, "valid", valid_transforms)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    mode = "train" if IS_RANGPUR else "debug"
    train_dataset = ProstateDataset3D(images_path, masks_path, mode, train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Model
    model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    model.to(device)
    model.apply(init_weights)
        
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Loss function
    criterion = DiceLoss(softmax=True, include_background=background)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, masks, affines = data
            inputs, masks = inputs.to(device), masks.to(device)


            one_hot_masks_3d = F.one_hot(masks, num_classes=6).permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            softmax_logits, predictions, logits = model(inputs) # All shapes: [batch * l * w * h, 6]
            
            loss = criterion(logits, one_hot_masks_3d)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
       
        scheduler.step()

        if epoch % 2 == 0:
            save(predictions, affines, epoch)

        if epoch % 3 == 0:
            validate(model, valid_dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}")

    # save model
    torch.save(model.state_dict(), f'model_lr_{lr}_e_{epochs}_bg_{background}_bs{batch_size}.pth')
    validate(model, valid_dataloader)