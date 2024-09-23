import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset, split_dataset
from utils import get_transform, train, test
from modules import GFNet
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/lcz/PatternAnalysis-2024/data/ADNI/AD_NC', type=str)
parser.add_argument('--show_progress', default="True", type=str)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--early_stopping', default=5, type=int)
parser.add_argument('--device', default="cuda", type=str)

args = parser.parse_args()

# define some global variables
device = args.device if torch.cuda.is_available() else "cpu"
num_epochs = args.epochs
batch_size = args.batch_size
early_stopping = args.early_stopping
disable_tqdm = not (args.show_progress == "True")

if __name__ == "__main__":
    
    # load the dataset
    train_dataset = ADNIDataset(root=args.data_path, split="train", transform=get_transform(train=True))
    
    # split dataset
    train_dataset, val_dataset = split_dataset(train_dataset, split_ratio=0.8, seed=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # create model
    model = GFNet(img_size=224, in_chans=1, num_classes=1, patch_size=14, embed_dim=256, 
                  depth=8, mlp_ratio=4, drop_path_rate=0.15, 
                  norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, disable_tqdm=disable_tqdm)
        test_loss, test_acc = test(model, val_loader, criterion, device, disable_tqdm=disable_tqdm)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    print('Training complete.')