'''
Containing the source code for training, validating, testing and saving trained model. 

Created by: Shogo Terashima
Created by:     Shogo Terashima
ID:             S47779628
Last update:    24/10/2024
'''

import torch
import os
import csv
from dataset import CombinedPreprocessing
from modules import GFNet
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
import torch.nn as nn
import torch.optim as optim
from utils import EarlyStopping
import optuna
from optuna.terminator import EMMREvaluator, MedianErrorEvaluator, Terminator, TerminatorCallback
import gc

def train_one_epoch(model, train_loader, criterion, optimiser, device):
    '''
    Train model one epoch with using train set and calculate train loss.
    '''
    model.train()
    running_loss = 0.0 
    for inputs, labels in train_loader:
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
    '''
    Use validation set and calculate loss (BCEWithLogitsLoss)
    '''
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    epoch_loss = running_loss / len(val_loader)
    return epoch_loss

def linear_warmup(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def objective(trial):
    '''
    Hyper parameter tuning with Optuna
    '''

    # Hyperparameters use for tuning and range of values
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2)    
    drop_rate = trial.suggest_float('dropout_rate', 0.0, 0.5) #dropout_rate
    drop_path_rate = trial.suggest_float('drop_path_rate', 0.0, 1.0, step=0.1)
    t_max = trial.suggest_int('tmax', 10, 50, log=False)

    # Not using for tuning
    batch_size = 64
    warmup_epochs = 5


    # Create folder based on hyperparameters to store model and csv file 
    folder_name = f"lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}"
    output_dir = os.path.join("experiments4", folder_name) 
    os.makedirs(output_dir, exist_ok=True)

    # Load train and validation data
    train_dataset_path = "/home/groups/comp3710/ADNI/AD_NC/train"
    test_dataset_path = "/home/groups/comp3710/ADNI/AD_NC/test"    
    seed = 20
    data_preprocessor = CombinedPreprocessing(
        train_path=train_dataset_path,
        test_path=test_dataset_path,
        batch_size=batch_size,
        num_workers=1,
        val_split=0.2,
        seed=seed
    )
    train_loader, val_loader, test_loader = data_preprocessor.get_data_loaders()

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # GFNet-H-B settings
    model = GFNet(
        image_size=224, 
        num_classes=1,
        blocks_per_stage=[3, 3, 27, 3], 
        stage_dims=[96, 192, 384, 768], 
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        init_values=1e-6
    )


    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_scheduler = LambdaLR(optimiser, lr_lambda=lambda epoch: linear_warmup(epoch, warmup_epochs))
    cosine_scheduler = CosineAnnealingLR(optimiser, T_max=t_max - warmup_epochs)
    checkpoint_path = os.path.join(output_dir, 'best_model.pt') 

    # scheduler settings. We perform warmup first, then apply cosine_scheduler
    scheduler = SequentialLR(
        optimiser, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )

    early_stopping = EarlyStopping(patience=5, min_delta=0.0001, path=checkpoint_path)

    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            best_val_loss = early_stopping.best_loss
            break
    

    print("Loading the best model from checkpoint.")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # Save losses
    loss_log_path = os.path.join(output_dir, 'loss_log.csv')
    with open(loss_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        for epoch_num, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch_num, t_loss, v_loss])

    print(f"Loss logs saved to {loss_log_path}")
    
    # Clear cache
    del model, optimiser, criterion, scheduler, early_stopping, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_loss



if __name__ == "__main__":
    # Set up optuna
    # early trial stopping settings
    emmr_improvement_evaluator = EMMREvaluator()
    median_error_evaluator = MedianErrorEvaluator(emmr_improvement_evaluator)
    terminator = Terminator(
        improvement_evaluator=emmr_improvement_evaluator,
        error_evaluator=median_error_evaluator,
    )
    terminator_callback = TerminatorCallback(terminator)

    # create and initialies optuna study
    study = optuna.create_study(
        direction='minimize', # I want to minimse the loss
        study_name='gfnet_study',
        storage='sqlite:///gfnet_study.db',
        load_if_exists=False
    )
    study.optimize(objective, n_trials=15, callbacks=[terminator_callback])

    print(f"Best Trial: {study.best_trial }")
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation loss: {study.best_value}")

    # Clear Cache
    del emmr_improvement_evaluator, median_error_evaluator, terminator, terminator_callback, study
    torch.cuda.empty_cache()
    gc.collect()

    