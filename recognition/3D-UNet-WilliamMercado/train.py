"""
train.py
"""
import os
from random import randrange
from statistics import fmean

import pandas as pd
import torch
from const import (ACCURACY_DATA_TARGET, DATASET_PATH, NET_OUTPUT_DIR,
                   NET_OUTPUT_TARGET)
from dataset import MriData3D, mri_split
from modules import FullUNet3D
from torch import nn, optim
from torch.utils.data import DataLoader
from util import UNIQUE_ROTATION_COMBOS, rotate

# CHECK CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("NO CUDA AVAILABLE. CPU IN USE")
print(device)

# Sum Training Options
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 1
NUM_CLASSES = 6

files_train, files_test, files_validate = mri_split(data_path=DATASET_PATH,proportions=[0.9, 0.05, 0.05])
# print(len(files_train + files_test + files_validate))

data_train = MriData3D(data_path=DATASET_PATH,target_data=files_train)
data_test = MriData3D(data_path=DATASET_PATH,target_data=files_test)
data_validate = MriData3D(data_path=DATASET_PATH,target_data=files_validate)

train_dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

if not os.path.isdir(NET_OUTPUT_DIR):
    os.mkdir(NET_OUTPUT_DIR)



if os.path.isfile(NET_OUTPUT_TARGET):
    model = torch.load(NET_OUTPUT_TARGET)
    model = model.to(device)
else:
    model = FullUNet3D(input_width=1,out_width=NUM_CLASSES).to(device=device)
    model = model.to(device)

if os.path.isfile(ACCURACY_DATA_TARGET):
    accuracy_data = pd.read_csv(ACCURACY_DATA_TARGET,index_col=0)
    overall_epoch = len(accuracy_data.index)
else:
    accuracy_data = None
    overall_epoch = 0

accuracy_store = []

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Start Training")
for epoch in range(NUM_EPOCHS):
    losses = []
    overall_epoch += 1
    model.train() # engage training mode
    for i, (images, labels) in enumerate(train_dataloader):
        images = torch.tensor(images,device=device)
        labels = torch.tensor(labels,device=device)
        b_size = images.size(0)
        print(images.shape)
        x, y, z = UNIQUE_ROTATION_COMBOS[randrange(len(UNIQUE_ROTATION_COMBOS))]
        for _ in range(x):
            images = rotate(images,2,3)
            labels = rotate(labels,2,3)
        for _ in range(y):
            images = rotate(images,2,4)
            labels = rotate(labels,2,4)
        for _ in range(z):
            images = rotate(images,3,4)
            labels = rotate(labels,3,4)

        # --- Train ---
        # Forward pass
        out:torch.Tensor = model(images)

        # Compute Error, backpropagate, optimize
        # out = out.to(device='cpu')
        calc_loss = loss(out.to(torch.float),labels.squeeze(dim=1).to(torch.long))
        calc_loss.backward()
        optimizer.step()

        # Training stats
        print('%d > [%d/%d][%d/%d]\tLoss: %.4f\tRot:(%d,%d,%d)'
                % (overall_epoch, epoch, NUM_EPOCHS, i, len(train_dataloader),
                calc_loss.item(),x,y,z), flush=True)

        # Save loss
        losses.append(calc_loss.item())
    # End Epoch
    avg_train_loss = fmean(losses)

    # Save epoch
    torch.save(model, NET_OUTPUT_TARGET)
    torch.save(model, NET_OUTPUT_DIR + f"net-checkpoint-{overall_epoch}")

    print(f"-- Test Loss For Epoch {overall_epoch} --")
    model.eval()
    test_losses = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_dataloader):
            images = torch.tensor(images,device=device)
            labels = torch.tensor(labels,device=device)

            # Forward pass
            out:torch.Tensor = model(images)

            # Compute Error, backpropagate, optimize
            # out = out.to(device='cpu')
            calc_loss = loss(out.to(torch.float),labels.squeeze(dim=1).to(torch.long))
            test_losses.append(calc_loss.item())
    avg_test_loss = fmean(test_losses)
    print(f"Average Train Loss: {avg_train_loss} \tAverage Test Loss:{avg_test_loss}")
    accuracy_store.append((avg_train_loss,avg_test_loss))


# Save final df
new_accuracy_data = pd.DataFrame(accuracy_store,columns=["Train Loss","Test Loss"])

if accuracy_data is not None:
    accuracy_data = pd.concat((accuracy_data,new_accuracy_data),ignore_index=True)
else:
    accuracy_data = new_accuracy_data

accuracy_data.to_csv(ACCURACY_DATA_TARGET)
