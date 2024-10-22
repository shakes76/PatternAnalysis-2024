from train import train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from tqdm import tqdm

model,test_loader = train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Print report
print(
    classification_report(
        y_true, y_pred, target_names=[f"Class {i}" for i in range(2)]
    )
)
