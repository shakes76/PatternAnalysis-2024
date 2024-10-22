from modules import AlzheimerModel
from dataset import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from torch import GradScaler, autocast
import os


from torch.optim.lr_scheduler import StepLR
import time
from sklearn.metrics import confusion_matrix, classification_report

# Parameters
START_EPOCH = 500
train_dir = '/home/groups/comp3710/ADNI/AD_NC/train' # '/home/groups/comp3710/ADNI/AD_NC/train'
test_dir = '/home/groups/comp3710/ADNI/AD_NC/test' # '/home/groups/comp3710/ADNI/AD_NC/train'

# Define hyperparameters and settings with minimal values
in_channels = 1
img_size = 224
patch_size = 16
embed_size = 768
num_layers = 12
num_heads = 8
d_mlp = 2048
dropout_rate = 0.4
num_classes = 2
batch_size = 32
learning_rate = 1e-5
weight_decay = 1e-4


if __name__ == "__main__":
    test_loader = create_test_loader(
        test_dir, 
        batch_size=batch_size, 
    )
    

    device = torch.device('cuda')
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")
    model = AlzheimerModel(in_channels, patch_size, embed_size, img_size, num_layers, num_heads, d_mlp, dropout_rate)
    model.to(device)

    checkpoint_path = f'output/param/checkpoint{START_EPOCH}.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'Loaded model from {checkpoint_path}')

    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(test_loader)

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
