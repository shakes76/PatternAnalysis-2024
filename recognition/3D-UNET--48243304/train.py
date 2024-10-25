import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset  
from modules import UNet3D  
import os
import torch.nn.functional as F

BATCH_SIZE = 1  
LEARNING_RATE = 0.001
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_dir = r"/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
label_data_dir = r"/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

train_dataset = Prostate3DDataset(data_dir=train_data_dir, label_dir=label_data_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet3D(in_channels=1, out_channels=1, init_features=32).to(device)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train() 
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device).float() 
            labels = labels.to(device).float()  

            # Forward pass
            outputs = model(images)
            
            # Resize outputs to match the labels' size if there's a mismatch
            if outputs.size() != labels.size():
                outputs = F.interpolate(outputs, size=labels.shape[2:], mode='trilinear', align_corners=False)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:  
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{total_batches}], '
                      f'Loss: {loss.item():.4f}, Running Loss: {running_loss/(batch_idx+1):.4f}')

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    print('Training completed.')

if __name__ == '__main__':
    train_model(model, train_loader, criterion, optimizer, EPOCHS)
    torch.save(model.state_dict(), 'unet3d_model.pth')

for images, labels in train_loader:
    images = images.to(device).float()  # Ensure images have shape (batch_size, 1, depth, height, width)
    labels = labels.to(device).float()  # Ensure labels have shape (batch_size, 1, depth, height, width)
    
    print(f'Images shape: {images.shape}')  
    outputs = model(images)  # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
