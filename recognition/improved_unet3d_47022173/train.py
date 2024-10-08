from dataset import *
import torch
from torch.utils.data import DataLoader, random_split
import torchio as tio
from utils import *
from modules import *

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
images_path = "./data/semantic_MRs_anon/"
masks_path = './data/semantic_labels_anon/'
# images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/"
# masks_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/"

# Data parameters
batch_size = 2
shuffle = True
num_workers = 1
epochs = 5

# Model parameters
in_channels = 1 # greyscale
n_classes = 6 #6 different values in mask
base_n_filter = 8

# Optimizer parameters
lr = 1e-4
weight_decay = 1e-2

# Scheduler parameters
step_size = 10
gamma = 0.1

if __name__ == '__main__':
    # Load and process data
    transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.RandomFlip(),
        tio.Resize((128,128,128)),
        tio.RandomAffine(degrees=10),
        tio.RandomElasticDeformation(),
        tio.ZNormalization(),
    ])

    dataset = ProstateDataset3D(images_path, masks_path, transforms)
    fixed_gen = torch.Generator().manual_seed(SEED)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, 1 - TRAIN_SIZE], generator=fixed_gen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) #TODO Change to test split

    # Model
    model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    model.to(device)
    #INIT model weights
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Loss function
    criterion = diceLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            if masks.dtype != torch.long:
                masks = masks.long()

            optimizer.zero_grad()
            out, seg_layer, logits = model(inputs)
            masks = masks.view(-1)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
    
    # save model
    # torch.save(model.state_dict(), 'model.pth')