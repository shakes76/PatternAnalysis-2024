import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import UNet
from utils import load_data_2D
from modules import dice_loss
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# paths to the train, validation data
train_dir = r'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_train'
val_dir = r'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_validate'

# set hyperparams below
batch_size = 8
epochs = 50
learning_rate = 1e-4  # 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# load training and validation data using the load_data_2D function
train_images = load_data_2D([os.path.join(train_dir, f) for f in os.listdir(train_dir)])
val_images = load_data_2D([os.path.join(val_dir, f) for f in os.listdir(val_dir)])
train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False)

# Initialize model
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# binary cross-entropy loss for segmentation
criterion = nn.BCELoss()  

# lists to store losses and accuracy so they can be plotted later
train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []

# training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    dice_score = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data['image'].to(device), data['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # calculate both BCE and dice loss (using import from modules)
        bce_loss = criterion(outputs, labels)
        dice = dice_loss(outputs, labels)
        loss = bce_loss + dice
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        dice_score += 1 - dice.item()
    
    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = dice_score / len(train_loader)
    train_losses.append(avg_train_loss)
    train_dice_scores.append(avg_train_dice)

    # validation step using validation data
    model.eval()
    val_running_loss = 0.0
    val_dice_score = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(inputs)
            
            bce_loss = criterion(outputs, labels)
            dice = dice_loss(outputs, labels)
            loss = bce_loss + dice

            val_running_loss += loss.item()
            val_dice_score += 1 - dice.item()

    avg_val_loss = val_running_loss / len(val_loader)
    avg_val_dice = val_dice_score / len(val_loader)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Train Dice: {avg_train_dice}, Val Dice: {avg_val_dice}")

# plot the training and validation loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# plot the dice similarity coefficient (accuracy) for train and validation
plt.figure()
plt.plot(train_dice_scores, label='Train Dice Score')
plt.plot(val_dice_scores, label='Validation Dice Score')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.title('Training and Validation Dice Score')
plt.show()

# save the trained model for later use by test.py
torch.save(model.state_dict(), 'unet_prostate.pth')

# create confusion matrix
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        inputs, labels = data['image'].to(device), data['label'].to(device)
        outputs = model(inputs)
        
        # threshold the outputs to binary values (0 or 1)
        preds = (outputs > 0.5).float()
        
        all_preds.append(preds.cpu().view(-1).numpy())
        all_labels.append(labels.cpu().view(-1).numpy())

# flatten all predictions and labels
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Background', 'Prostate'], yticklabels=['Background', 'Prostate'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
