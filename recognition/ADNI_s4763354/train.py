import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('/home/Student/s4763354/comp3710/GFNet'))

from modules import GFNetBinaryClassifier
from dataset import get_data_loaders

def train_and_evaluate(train_dir, test_dir, epochs=10, lr=1e-4, batch_size=32):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size=batch_size)
    # # Fetch a single batch from the training loader
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    
    # print(f'Image batch shape: {images.shape}')  # [batch_size, 3, 224, 224]
    # print(f'Label batch shape: {labels.shape}')  # [batch_size]
    # print(f'First label in the batch: {labels[0]}') 

    # Initialize model, loss, and optimizer
    model = GFNetBinaryClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


if __name__ == "__main__":
    train_and_evaluate(train_dir='/home/groups/comp3710/ADNI/AD_NC/train', 
                    test_dir='/home/groups/comp3710/ADNI/AD_NC/test', epochs=1)
