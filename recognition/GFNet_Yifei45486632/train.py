import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import CustomImageDataset
from modules import GFNet, GFNetPyramid
from functools import partial
import os
import numpy as np
import kornia

# 定义设备
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def auto_crop_black_edges(image, threshold=5):
    """
    Adaptively crop the black background of the image.

    Parameters:
    -image: PIL Image object
    -threshold: This is the threshold of the gray value below which the region is considered black

    Returns:
    - Cropped PIL Image object
    """
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    mask = np_image > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    return cropped_image

# Define the dataset catalog and transformation
directory = "./train"
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True)

# Define the data augmentation transformation
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: auto_crop_black_edges(img)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: kornia.enhance.equalize_clahe(img, clip_limit=2.0, grid_size=(8,8))),  # CLAHE 增强
    transforms.Lambda(lambda img: kornia.enhance.adjust_gamma(img, gamma=1.2)),  # Gamma 调整
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: auto_crop_black_edges(img)),
    transforms.Resize((224, 224)),     
    transforms.ToTensor(),
    transforms.Lambda(lambda img: kornia.enhance.equalize_clahe(img, clip_limit=2.0, grid_size=(8,8))),  # CLAHE 增强
    transforms.Lambda(lambda img: kornia.enhance.adjust_gamma(img, gamma=1.2)),  # Gamma 调整
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Instantiate the dataset
dataset = CustomImageDataset(directory='./train')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Define the data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Instantiate a model
model = GFNet(
    img_size=224, 
    patch_size=16, in_chans=3, num_classes=2,
    embed_dim=256, depth=6, mlp_ratio=4, drop_path_rate=0.15,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training and validation
num_epochs = 50
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(100. * correct / total)
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100. * correct / total)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")
    
    # Save the model for each epoch
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path} at epoch {epoch+1}")

    # Update the learning rate scheduler
    scheduler.step()

# Plot the loss and accuracy curves
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
