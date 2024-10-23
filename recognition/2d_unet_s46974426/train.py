import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import UNet
from utils import load_data_2D
from modules import dice_loss
import os

# paths to the train, test, validation data
train_dir = r'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_train'
val_dir = r'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_validate'

# hyperparameters
batch_size = 8
epochs = 50
learning_rate = 1e-4 # 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# load training data
train_images = load_data_2D([os.path.join(train_dir, f) for f in os.listdir(train_dir)])
train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)

# initialize model
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for segmentation

# training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data['image'].to(device), data['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate both BCE and Dice loss
        bce_loss = criterion(outputs, labels)
        dice = dice_loss(outputs, labels)
        loss = bce_loss + dice
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
