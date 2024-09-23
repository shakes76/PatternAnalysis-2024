from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import ADNIDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import GFNet
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/lcz/PatternAnalysis-2024/data/ADNI/AD_NC', type=str)
parser.add_argument('--show_progress', default=True, type=bool)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--early_stopping', default=5, type=int)

args = parser.parse_args()

def get_transform(train):
    transform_list = [
        transforms.ToTensor(),
    ]
    if train:
        transform_list.extend([
            transforms.RandomRotation(10),  # rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # adjust brightness and contrast
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Gaussian Blur
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # Random Erasing
            transforms.RandomApply([transforms.ElasticTransform()], p=0.5)  # Elastic Transform
        ])
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(transform_list)

print(os.path.join(args.data_path, 'train'))

train_dataset = ADNIDataset(root=os.path.join(args.data_path, 'train'), transform=get_transform(train=True))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = ADNIDataset(root=os.path.join(args.data_path, 'test'), transform=get_transform(train=False))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


model = GFNet(
            img_size=210, in_chans=1, num_classes=1,
            patch_size=7, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )


# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

# Training and testing loop
num_epochs = 30  # Adjust number of epochs as needed

def train(model, train_loader, optimizer, criterion, use_tqdm=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, disable=use_tqdm):
        inputs, labels = inputs.to(device), labels.float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def test(model, test_loader, criterion, use_tqdm=True):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, disable=use_tqdm):
            inputs, labels = inputs.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Statistics
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, use_tqdm=args.show_progress)
    test_loss, test_acc = test(model, test_loader, criterion, use_tqdm=args.show_progress)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

print('Training complete.')
