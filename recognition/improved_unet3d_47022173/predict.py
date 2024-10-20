"""
This file contains the code to load a trained model and run predictions on the test set. 
"""

import torch
from dataset import *
import torch
from torch.utils.data import DataLoader, random_split
import torchio as tio
from utils import *
from modules import *
from monai.losses.dice import DiceLoss
from train import *

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

if IS_RANGPUR:
    images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/"
    masks_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/"
    epochs = 50
    batch_size = 4
else:
    images_path = "./data/semantic_MRs_anon/"
    masks_path = "./data/semantic_labels_anon/"
    epochs = 5
    batch_size = 2

# Model parameters
in_channels = 1 # greyscale
n_classes = 6 # 6 different values in mask
base_n_filter = 8

batch_size = batch_size
num_workers = 2

if __name__ == '__main__':
    test_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.Resize((128,128,128)),
        tio.ZNormalization(),
    ])
    
    test_dataset = ProstateDataset3D(images_path, masks_path, "test", test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    

    model.eval()
    test_loss = 0.0
    dice_scores = [0] * n_classes 
    criterion = DiceLoss()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):  
            inputs, masks, affines = data
            inputs, masks = inputs.to(device), masks.to(device)
            one_hot_masks_3d = F.one_hot(masks, num_classes=6).permute(0, 4, 1, 2, 3)
            softmax_logits, predictions, logits = model(inputs) # All shapes: [batch * l * w * h, 6]
            save(predictions, affines, i)


    # Dice score
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Average Dice Score: {list(map(lambda x: float(x / len(test_dataloader)), dice_scores))}")