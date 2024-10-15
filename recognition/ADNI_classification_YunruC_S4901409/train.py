import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from modules import GFNetClassifier
from dataset import get_data_loaders

def train(model, loader, criterion, optimizer, device, clip_value = 1.0):
    model.train()
    total_loss= 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion (outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return total_loss/len(loader), accuracy_score(all_labels, all_preds)



