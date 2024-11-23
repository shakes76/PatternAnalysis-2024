"""
This file contains the main code for training, validating, and saving the UNet model,
as well as plotting the training losses and validation dice score.
"""

from dataset import load_data_2D
from params import *
from modules import *

import torch
import time
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Data Pre-processing

# Normalise images
train_imgs = sorted([os.path.join(TRAIN_IMG_DIR, img) for img in os.listdir(TRAIN_IMG_DIR) if img.endswith(('.nii', '.nii.gz'))])
train_imgs = load_data_2D(train_imgs, True)

validation_imgs = sorted([os.path.join(VAL_IMG_DIR, img) for img in os.listdir(VAL_IMG_DIR) if img.endswith(('.nii', '.nii.gz'))])
validation_imgs = load_data_2D(validation_imgs, True)

# One-hot encode the labels
train_labels = sorted([os.path.join(TRAIN_MASK_DIR, img) for img in os.listdir(TRAIN_MASK_DIR) if img.endswith(('.nii', '.nii.gz'))])
train_labels = load_data_2D(train_labels, False, True)

validation_labels = sorted([os.path.join(VAL_MASK_DIR, img) for img in os.listdir(VAL_MASK_DIR) if img.endswith(('.nii', '.nii.gz'))])
validation_labels = load_data_2D(validation_labels, True)

# Create dataloaders
training_set = [ [train_imgs[i], train_labels[i]] for i in range(len(train_imgs))]
train_loader = torch.utils.data.DataLoader(training_set, BATCH_SIZE, True)

validation_set = [ [validation_imgs[i], validation_labels[i]] for i in range(len(validation_imgs))]
val_loader = torch.utils.data.DataLoader(validation_set, BATCH_SIZE, True)

# Initialise the model
model = UNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reduce learning rate by a factor of 0.1 if mean epoch loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

# Training loop
model.train()
start = time.time()
losses = [] # store all training losses
dice_scores = [] # store dice score calculated on validation set
for epoch in range(NUM_EPOCHS):
    
    avg_epoch_losses = [] # store avg training loss for each epoch

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        epoch_loss = [] # store training losses for the current epoch 

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images.unsqueeze(dim=0))
        loss = loss_function(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epoch_loss.append(loss.item())

        if (batch_idx+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, NUM_EPOCHS, batch_idx+1, len(train_loader), loss.item()))
    
    # Calculate average loss
    avg_epoch_losses.append(np.mean(epoch_loss))

    # Update learning rate
    scheduler.step(avg_epoch_losses[-1])

    # calculate dice score on validation set
    model.eval()
    dice_score = dice_score(val_loader, device, model)
    print(f'Validation dice score: {dice_score}')
    model.train()

    # Save a checkpoint after each epoch
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename=f'epoch_data/epoch_{epoch}/checkpoints/checkpoint.pth.tar')
        
# Save checkpoint after training is complete
checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}
torch.save(checkpoint, filename=CHECKPOINT_DIR)

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Plot training losses
plt.figure(figsize=(20, 10))
plt.plot(losses, label='Loss')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.grid(True)
plt.savefig('plots/training_losses.png')
#plt.show()

# plot Average Losses per Epoch
plt.figure(figsize=(20, 10))
plt.plot(avg_epoch_losses, label='Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title('Average Losses per Epoch')
plt.grid(True)
plt.savefig('plots/avg_epoch_losses.png')
#plt.show()
    
# plot Dice score for each epoch
plt.figure(figsize=(20, 10))
plt.plot(dice_scores, label='Dice Score')
plt.xlabel('Epoch #')
plt.ylabel('Dice Score')
plt.title('Validation Dice Scores')
plt.grid(True)
plt.savefig('plots/dice_scores.png')
#plt.show()