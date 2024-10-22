import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_dataloaders
from tqdm import tqdm
from modules import TransformerNet
import os
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import classification_report


def train_model(epochs=10):
    if os.path.exists("recognition/classify_azheimer/AD_NC"):
        data_dir = "recognition/classify_azheimer/AD_NC"
    else:
        data_dir = "/home/groups/comp3710/ADNI/AD_NC/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNet(num_classes=2) # 2 classes: AD and NC
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    train_loader, test_loader = get_dataloaders(data_dir)
    bar = tqdm(range(epochs * len(train_loader)))
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            bar.update(1)
        scheduler.step()  # Update the learning rate
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    torch.save(model.state_dict(), "alzheimer_classifier.pth")
    return model
