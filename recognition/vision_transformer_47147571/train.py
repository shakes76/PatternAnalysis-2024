"""
Train and save the model.
The model will be saved to `recognition/vision_transformer_47147571/logs/GFNet/best_gfnet.pt`, 
and a corresponding TensorBoard event will be generated in the same directory.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ADNIDataset, ADNIDatasetTest
from utils import get_transform, train, test, set_seed
from modules import GFNet
from functools import partial
import copy
import numpy as np
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/lcz/PatternAnalysis-2024/data/ADNI/AD_NC', type=str)
parser.add_argument('--show_progress', default="True", type=str)
parser.add_argument('--epochs', default=60, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--early_stopping', default=20, type=int)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--train_seed', default=1, type=int)

args = parser.parse_args()

# fix training seed and define some global variables
set_seed(args.train_seed)
device = args.device if torch.cuda.is_available() else "cpu"
num_epochs = args.epochs
batch_size = args.batch_size
early_stopping = args.early_stopping
disable_tqdm = not (args.show_progress == "True")

script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, 'logs/GFNet')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    
    # load the dataset
    train_dataset = ADNIDataset(root=args.data_path, split="train", transform=get_transform(train=True), 
                                val=False, seed=0, split_ratio=0.9)
    
    val_dataset = ADNIDataset(root=args.data_path, split="train", transform=get_transform(train=False), 
                              val=True, seed=0, split_ratio=0.9)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # create model
    # gfnet-xs
    model = GFNet(img_size=210, in_chans=1, patch_size=14, embed_dim=384, depth=12, mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    best_val_acc = 0
    patience_counter = 0 # count patience for early stopping
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, device, disable_tqdm=disable_tqdm)
        val_loss, val_acc = test(model, val_loader, criterion, device, disable_tqdm=disable_tqdm)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        
        # Save training loss and acc
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(log_dir, 'best_gfnet.pt'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
    print('Training complete.')
    
    # test
    print('-'*15)
    print('Testing...')
    test_dataset = ADNIDatasetTest(root=args.data_path, transform=get_transform(train=False))
    
    best_model.eval()
    preds_list = []
    true_list = []

    # test
    for data, labels in test_dataset:
        data, labels = data.to(device), labels.float().to(device)
        
        outputs = best_model(data)
        
        outputs = torch.sigmoid(outputs).mean().item()
        preds = 1 if outputs > 0.5 else 0

        preds_list.append(preds)
        true_list.append(labels.item())
        
    preds_list = np.array(preds_list)
    true_list = np.array(true_list)

    # Calculate accuracy
    accuracy = accuracy_score(true_list, preds_list)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")