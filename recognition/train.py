# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from module import SiameseNetwork
from dataset import SiameseISICDataset
import random
import numpy as np

# Set random seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Set hyperparameters
BATCH_SIZE = 8
NUMBER_EPOCHS = 100
IMG_SIZE = 100

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Create datasets and data loaders
train_dataset = SiameseISICDataset(image_folder='/home/groups/comp3710/ISIC2018/train', transform=transform)
val_dataset = SiameseISICDataset(image_folder='/home/groups/comp3710/ISIC2018/val', transform=transform)

trainloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=BATCH_SIZE)
valloader = DataLoader(val_dataset, shuffle=False, num_workers=8, batch_size=BATCH_SIZE)

# Define model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SiameseNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(NUMBER_EPOCHS):
    print(f"Epoch {epoch+1}/{NUMBER_EPOCHS} start.")
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        img0, img1, labels = data
        img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)
        labels = labels.long()

        optimizer.zero_grad()
        outputs = net(img0, img1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if i % 10 == 0:
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss.item())
    
    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{NUMBER_EPOCHS}], Loss: {epoch_loss:.4f}")
    
    # Validate the model
    net.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data in valloader:
            img0, img1, labels = data
            img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)
            labels = labels.long()
            outputs = net(img0, img1)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    accuracy = 100 * correct_val / total_val
    print(f'Validation Accuracy: {accuracy:.2f}%')
    
    # Visualize loss curve (if needed)
    # show_plot(counter, loss_history)
