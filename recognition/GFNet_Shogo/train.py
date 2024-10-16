'''
Containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
Make sure to plot the losses and metrics during training

Created by: Shogo Terashima
'''

import torch
import os
import csv
from dataset import TrainPreprocessing, TestPreprocessing
from modules import GFNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import EarlyStopping
import optuna


# Learning Rate: 1e-5 to 1e-2
# Weight Decay: 1e-5 to 1e-2
# Dropout Rate: 0.0 to 0.5
# Drop Path Rate: 0.0 to 0.5
# Batch Size: 16, 32, 64

def objective(trial):
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('wd', 1e-5, 1e-2, log=True)
    drop_rate = trial.suggest_float('drop', 0.0, 0.5)
    drop_path_rate = trial.suggest_float('droppath', 0.0, 0.5)
    batch_size = trial.suggest_categorical('bs', [16, 32, 64, 128, 256])

    config = {
        'lr': learning_rate,
        'wd': weight_decay,
        'drop': drop_rate,
        'droppath': drop_path_rate,
        'bs': batch_size
    }

    # Create folder based on hyper parameters
    folder_name = f"lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}"
    output_dir = os.path.join("experiments", folder_name) 
    os.makedirs(output_dir, exist_ok=True)

    # load data
    train_dataset_path = "../dataset/AD_NC/train"

    # Load train and validation data
    train_data = TrainPreprocessing(train_dataset_path, batch_size=batch_size)
    train_loader, val_loader = train_data.get_train_val_loaders(val_split=0.2)

    # model initalisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test model (tiny tiny)
    model = GFNet(
        img_size=224, 
        num_classes=1,
        initial_embed_dim=32, 
        blocks_per_stage=[1, 1, 1, 1], 
        stage_dims=[32, 64, 128, 256], 
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        init_values=1e-5
    )
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=100) 
    checkpoint_path = os.path.join(output_dir, 'best_model.pt') 
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, path=checkpoint_path)

    # Training loop
    num_epochs = 1
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # train
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        scheduler.step()
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    print("Loading the best model from checkpoint.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Reporting lossess
    loss_log_path = os.path.join(output_dir, 'loss_log.csv')
    with open(loss_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        for epoch_num, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch_num, t_loss, v_loss])

    print(f"Loss logs saved to {loss_log_path}")
    return val_loss

def train_one_epoch(model, train_loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0 
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)  
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    epoch_loss = running_loss / len(val_loader)
    return epoch_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation loss: {study.best_value}")