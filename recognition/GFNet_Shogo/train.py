'''
Containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
Make sure to plot the losses and metrics during training

Created by: Shogo Terashima
'''

import torch
from dataset import TrainPreprocessing, TestPreprocessing
from modules import GFNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib as plt


# Learning Rate: 1e-5 to 1e-2
# Weight Decay: 1e-5 to 1e-2
# Dropout Rate: 0.0 to 0.5
# Drop Path Rate: 0.0 to 0.5
# Batch Size: 16, 32, 64


# path to datasets
train_dataset_path = "../dataset/AD_NC/train"
test_dataset_path = "../dataset/AD_NC/test"

# Load train and validation data
train_data = TrainPreprocessing(train_dataset_path, batch_size=128)
train_loader, val_loader = train_data.get_train_val_loaders(val_split=0.2)

# Load test data
test_data = TestPreprocessing(test_dataset_path, batch_size=16)
test_loader = test_data.get_test_loader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initial settings
test_model = GFNet(
    img_size=224, 
    num_classes=1,
    initial_embed_dim=32, 
    blocks_per_stage=[1, 1, 1, 1], 
    stage_dims=[32, 64, 128, 256], 
    drop_rate=0.05,
    drop_path_rate=0.05,
    init_values=1e-5,
    is_training=True
)

test_model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimiser = optim.AdamW(test_model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=100) 
torch.nn.utils.clip_grad_norm_(test_model.parameters(), max_norm=1.0)

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