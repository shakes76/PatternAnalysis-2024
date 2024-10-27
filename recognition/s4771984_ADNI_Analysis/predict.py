import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from modules import GFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation of the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return all_preds, all_labels, avg_loss

# Plotting the confusion matrix
def plot_confusion_matrix(labels, preds):
    conf_matrix = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

