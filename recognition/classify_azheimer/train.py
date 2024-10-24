import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_dataloaders
from tqdm import tqdm
from modules import GFNetPyramid
import os
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt 

def train_model(epochs=10):
    if os.path.exists("recognition/classify_azheimer/AD_NC"):
        data_dir = "recognition/classify_azheimer/AD_NC"
    else:
        os.environ['MPLCONFIGDIR'] = './.matplotlib_cache'
        data_dir = "/home/groups/comp3710/ADNI/AD_NC/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GFNetPyramid(num_classes=2)  # 2 classes: AD and NC
    # model = torch.load("alzheimer_classifier.pth")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.975)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.3)

    train_loader, test_loader = get_dataloaders(data_dir,128)
    bar = tqdm(range(epochs * len(train_loader)))
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            bar.update(1)
        scheduler.step()  # Update the learning rate
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs + 1), epoch_losses, color='red', label='Loss')
    plt.plot(range(1, epochs + 1), epoch_accuracies, color='blue', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_plot.png")
    torch.save(model.state_dict(), "./alzheimer_classifier.pth")
    return model
