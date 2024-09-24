import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ADNIDataset
from utils import get_transform, train, test, set_seed
from modules import GFNet
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/lcz/PatternAnalysis-2024/data/ADNI/AD_NC', type=str)
parser.add_argument('--show_progress', default="True", type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--early_stopping', default=30, type=int)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--train_seed', default=0, type=int)

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
                                val=False, seed=0, split_ratio=0.8)
    
    val_dataset = ADNIDataset(root=args.data_path, split="train", transform=get_transform(train=False), 
                              val=True, seed=0, split_ratio=0.8)
    
    test_dataset = ADNIDataset(root=args.data_path, split="test", transform=get_transform(train=False), 
                               val=False, split_ratio=1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # create model
    # gfnet-ti
    model = GFNet(img_size=210, in_chans=1, patch_size=15, embed_dim=256, 
                  depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    best_val_acc = 0
    patience_counter = 0 # count patience for early stopping
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, disable_tqdm=disable_tqdm)
        val_loss, val_acc = test(model, val_loader, criterion, device, disable_tqdm=disable_tqdm)
        test_loss, test_acc = test(model, test_loader, criterion, device, disable_tqdm=disable_tqdm)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        # Save training loss and acc
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_gfnet.pt'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
    print('Training complete.')