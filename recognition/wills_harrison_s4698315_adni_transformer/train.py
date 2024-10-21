import  torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from modules import SwinTransformer

import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import os

from predict import accuracy


def make_model():
    model = SwinTransformer(
        img_size=224, patch_size=4, in_chans=1, num_classes=2,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    return model

def train(patience=15, lr=1e-4, batch_size=64, show_plots=False, 
          model_type='gfnet-big', num_epochs=20, weight_decay=0.01,
          save_model=True):
    
    print(f"Training with parameters {lr} {batch_size} {weight_decay}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device {device}")
    
    train_ad_path = './train/AD'
    train_nc_path = './train/NC'
    
    test_ad_path = './test/AD'
    test_nc_path = './test/NC'
    
    
    
    train_dataset = ADNIDataset(train_ad_path, train_nc_path)
    test_dataset = ADNIDataset(test_ad_path, test_nc_path, transform=False)
    
    # split test dataset into validation and test
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = make_model(model_type)
    
    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # cosine scheduler
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # Lists to store loss values
    loss_values = []
    val_loss_values = []

    # Early stopping parameters
    best_loss = np.inf
    epochs_no_improve = 0
    early_stop = False
    best_model = None

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break
            
        # Training loop
        model.train()
        running_loss = 0.0
        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}'):
            images = images.to(device).float()  # Ensure images are float32
            labels = labels.to(device).long()   # Ensure labels are int64
            
            preds = model(images)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()


        avg_train_loss = running_loss / len(train_loader)
        loss_values.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode for validation
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device).float()
                labels = labels.to(device).long()

                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_loss_values.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        # print(f'Validation Accuracy: {accuracy(model)}')
        
        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs due to no improvement.")
            early_stop = True

        # Step the learning rate scheduler after each epoch
        # lr_scheduler.step()
        
    # Plot the training and validation loss curves
    if show_plots:
        plt.plot(loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.title('Training and Validation Loss over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        
    model = make_model(model_type)
    model.load_state_dict(best_model)
    model.to(device)
    acc = accuracy(model, val_loader)
    print(f'Validation Accuracy: {acc}')
    
    test_acc = accuracy(model, test_loader)
    print(f'Test Accuracy: {test_acc}')
    
    
    # create model directory
    if not os.path.exists('models'):
        os.makedirs('models')

    
    # Save the model
    if save_model:
        torch.save(best_model, f'models/{model_type}_{lr}_{weight_decay}_{batch_size}.pth')

def main():
    parameters = {
        'lr': [1e-5],
        'batch_size': [32],
        'weight_decay': [0.005]
    }
    
    
    # Generate all combinations of parameters
    param_values = [v for v in parameters.values()]
    param_combinations = list(itertools.product(*param_values))
    model_type = 'swin'
    
    for lr, batch_size, weight_decay in param_combinations:
        
        # check if the model has already been trained
        if not os.path.exists(f'models/{model_type}_{lr}_{weight_decay}_{batch_size}.pth'):
            train(lr=lr, batch_size=batch_size, model_type=model_type, show_plots=False,
                  num_epochs=100, weight_decay=weight_decay, save_model=True)
            
        else:
            print(f"Model with parameters {lr} {batch_size} {weight_decay} already trained.")
            
        
    
    

if __name__ == '__main__':
    main()