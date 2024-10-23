# train.py
import torch
import torch.nn as nn
from modules import UNet3D
from tqdm import tqdm
import time 
import os

os.environ['TQDM_DISABLE'] = 'True'

# Define Dice Loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        outputs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        labels = labels.float()  # Ensure labels are float for computation

        # Flatten the tensors
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        intersection = (outputs * labels).sum()
        dice_score = (2. * intersection + self.smooth) / (outputs.sum() + labels.sum() + self.smooth)
        
        return 1 - dice_score  # Return Dice Loss (1 - Dice coefficient)
    
class Trainer:
    def __init__(self, train_loader):
        # Initialize the 3D U-Net model
        self.model = UNet3D(in_channels=4, out_channels=6, init_features=32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Define optimizer and loss function
        self.criterion = DiceLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # Set the training data loader
        self.train_loader = train_loader

    def train(self, n_epochs=1):
        for epoch in range(n_epochs):
            start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{n_epochs}:")
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(self.train_loader, disable = True):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # Zero the gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_time = time.time() - start_time
                running_loss += loss.item()
            print(f"Epoch {epoch + 1} completed - Loss: {running_loss:.4f}, Time Taken: {epoch_time:.2f} seconds")
        
        return self.model
