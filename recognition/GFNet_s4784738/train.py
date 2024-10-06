"""
Trains, validates, tests, and saves a GFNet image classification model.
The model classifies brain images according to whether or not the subject has Alzheimer's disease

Benjamin Thatcher 
s4784738    
"""

# train.py
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from modules import GFNetModel
from dataset import get_data_loader
import matplotlib.pyplot as plt

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    model.to(device)
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimization only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

        print()

    torch.save(model.state_dict(), 'gfnet_alzheimer_model.pth')

    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    # Specify the correct paths to your train and validation sets
    dataloaders = {
        'train': get_data_loader("/home/groups/comp3710/ADNI/train", batch_size=32, shuffle=True),
        'val': get_data_loader("/home/groups/comp3710/ADNI/test", batch_size=32, shuffle=False)
    }

    model = GFNetModel(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

