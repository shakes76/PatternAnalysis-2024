import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from modules import GFNet
from dataset import get_data_loaders
from utils import draw_training_log


def validate(model, test_loader, criterion, device):
    """
    Validate the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def final_validate(model, test_loader, criterion, device):
    """
    Validate the test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append true and predicted labels for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    cm = confusion_matrix(all_labels, all_preds)

    TP = cm[1, 1]  # True Positives
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"test Loss: {test_loss:.4f}, test Acc: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GFNet(img_size=224, patch_size=16, in_chans=1, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    if os.path.exists('gfnet_model_latest.pth'):
        load_model(model, optimizer, 'gfnet_model_latest.pth')
        criterion = nn.CrossEntropyLoss()

        train_dir = 'new_store_data_path/AD_NC_new/train'
        val_dir = 'new_store_data_path/AD_NC_new/val'
        test_dir = 'new_store_data_path/AD_NC_new/test'

        train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)
        model = model.to(device)

        final_validate(model, test_loader, criterion, device)
    else:
        print("Can't find model pth")

    if os.path.exists('training_log.csv'):
        draw_training_log()
    else:
        print("Can't find training log")
