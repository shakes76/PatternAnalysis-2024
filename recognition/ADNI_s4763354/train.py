'''
Author: Lok Yee Joey Cheung
This file contains the functions of GFNet training, validating and testing processes, with visualizations on train and val loss and accuracy.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial

from modules import GFNet, GFNetPyramid, create_gfnet_pyramid, create_gfnet
from dataset import get_data_loaders

def train_and_evaluate(train_dir, test_dir, epochs=50, lr=5e-6, batch_size=64, pretrained=False, model_name='gfnet_h_b',pretrained_model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size=batch_size)

    # Define the model
    model = create_gfnet_pyramid(model_name=model_name)

    # If using approach 2, Load the state dict and extract the model weights of pre-trained model
    if pretrained and pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    
    #Modify the final layer for binary classification
    num_features = model.head.in_features 
    model.head = nn.Sequential(
        nn.Dropout(0.7),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(512, 2)
    )
    
    model = model.to(device)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the head (always trainable)
    for param in model.head.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  

    # Initialize the learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True,min_lr=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0
    patience = 5
    no_improve = 0

    # Define unfreezing stages
    unfreezing_stages = [
        model.blocks[-1],  # Last block
        model.blocks[-2],  # Second to last block
        model.blocks[-3],  # Third to last block
        model.blocks[:-3]  # All remaining blocks
    ]
    epochs_per_stage = epochs // (len(unfreezing_stages) + 1)  # +1 for initial training of just the head
    stage = 0

    actual_epochs = 0
    for epoch in range(epochs):
        # Unfreeze next stage 
        if epoch % epochs_per_stage == 0 and stage < len(unfreezing_stages):
            print(f"Unfreezing stage {stage + 1}")
            for param in unfreezing_stages[stage].parameters():
                param.requires_grad = True
            stage += 1
            # Reset optimizer and scheduler for the new set of trainable parameters
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - epoch, eta_min=1e-6)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Compute training accuracy and loss
        train_accuracy = 100 * correct / total
        average_train_loss = running_loss / len(train_loader)
        train_losses.append(average_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)

        actual_epochs += 1
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'TEST.pth')
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Final Train Loss: {average_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Final Val Loss: {average_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            break

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {average_train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {average_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')
        
        # Step the scheduler at the end of the epoch
        scheduler.step(val_accuracy)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr}')

    # Load the best model for testing
    model.load_state_dict(torch.load('TEST.pth', weights_only=True))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_TEST.png") 

    # Plotting training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, actual_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acc_TEST.png")  

    # torch.save(model.state_dict(), 'gfnet-h-b_50.pth')

if __name__ == "__main__":

    # For using the gfnet-h-b architecture with a specific pretrained model
    train_and_evaluate(
        train_dir='/home/groups/comp3710/ADNI/AD_NC/train', 
        test_dir='/home/groups/comp3710/ADNI/AD_NC/test',
        epochs=50,
        lr=5e-6,
        batch_size=64,
        pretrained=True,
        model_name='gfnet_h_b',
        pretrained_model_path='gfnet-h-b.pth'  
    )

    # For using another architecture without a pretrained model
    # train_and_evaluate(
    #     train_dir='/home/groups/comp3710/ADNI/AD_NC/train', 
    #     test_dir='/home/groups/comp3710/ADNI/AD_NC/test',
    #     epochs=50,
    #     lr=5e-6,
    #     batch_size=64,
    #     pretrained=False,
    #     model_name='gfnet-h-ti'
    # )
