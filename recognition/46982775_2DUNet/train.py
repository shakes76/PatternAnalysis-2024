# containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

""" 
Modified from Shekhar "Shakes" Chandra:
https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing#scrollTo=w2QhUgaco7Sp

"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from dataset import ProstateDataset
from modules import UNet

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-params
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 2

# Model name - Change for each training
model_name = "2d_unet_initial"
model_checkpoint1 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', model_name, 'checkpoint1.pth')
model_checkpoint2 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', model_name, 'checkpoint2.pth')
model_checkpoint3 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', model_name, 'checkpoint3.pth')
model_checkpoint4 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', model_name, 'checkpoint4.pth')

# File paths
main_dir = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data')
train_image_path = os.path.join(main_dir, 'keras_slices_train')
train_mask_path = os.path.join(main_dir, 'keras_slices_seg_train')
test_image_path = os.path.join(main_dir, 'keras_slices_test')
test_mask_path = os.path.join(main_dir, 'keras_slices_seg_test')

# Saving a model
# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
# EPOCH = 5
# PATH = "model.pt"
# LOSS = 0.4

# torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)
# checkpoint = torch.load(PATH, weights_only=True)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# Datasets
train_dataset = ProstateDataset(train_image_path, train_mask_path)

# Dataloader
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

# Model
model = UNet(in_channels=1, out_channels=6, n_features=64)
model = model.to(device)

# Loss Function and Optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# def train_model(model, loader, criterion, optimiser):
model.train()
start = time.time() #time generation
train_losses = []
for epoch in range(NUM_EPOCHS):
    epoch_running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = torch.unsqueeze(images, dim=1)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f" Step: {i+1}/{len(train_loader)}, Running Loss: {loss.item()}")
    epoch_loss = epoch_running_loss / len(train_loader)
    print(f" Epoch: {epoch}/{NUM_EPOCHS}, Epoch Loss: {epoch_loss}")
    train_losses.append(epoch_loss)

    # Save models
    if epoch + 1 == NUM_EPOCHS / 4:
        print("Saving first checkpoint")
        torch.save(model.state_dict(), model_checkpoint1)
    if epoch + 1 == NUM_EPOCHS / 2:
        print("Saving second checkpoint")
        torch.save(model.state_dict(), model_checkpoint2)
    if epoch + 1 == (3 * NUM_EPOCHS / 4):
        print("Saving third checkpoint")
        torch.save(model.state_dict(), model_checkpoint3)
    if epoch + 1 == NUM_EPOCHS:
        print("Saving fourth checkpoint (final model)")
        torch.save(model.state_dict(), model_checkpoint4)

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")



# Datasets
test_dataset = ProstateDataset(test_image_path, test_mask_path)

# Dataloader
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

# Test the model
start = time.time() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f" Step: {i+1}/{len(test_loader)}, Accuracy: {correct / total * 100}%")

    print('Final test Accuracy: {} %'.format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")