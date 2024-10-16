"""
train.py
"""
import numpy as np
from const import DATASET_PATH
from dataset import mri_split, MriData3D
from modules import FullUNet3D
from torch.utils.data import DataLoader
import torch

# CHECK CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("NO CUDA AVAILABLE. CPU IN USE")

files_train, files_test, files_validate = mri_split(data_path=DATASET_PATH,proportions=[0.7, 0.2, 0.1])
print(len(files_train + files_test + files_validate))

data_train = MriData3D(data_path=DATASET_PATH,target_data=files_train)
data_test = MriData3D(data_path=DATASET_PATH,target_data=files_test)
data_validate = MriData3D(data_path=DATASET_PATH,target_data=files_validate)

train_dataloader = DataLoader(data_train, batch_size=10, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=10, shuffle=True)

num_epochs = 5

model = FullUNet3D().to(device=device)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = torch.tensor(images,device=device)
        labels = torch.tensor(labels,device=device)
        b_size = images.size(0)
        print(images.shape)

        # --- Train ---
        D.zero_grad()
        # Forward pass D reals
        D_outputs = D(images).view(-1)
        D_loss_real = criterion(D_outputs, torch.ones_like(D_outputs))
        D_x = D_outputs.mean().item()

        # Forward pass D generated
        noise = torch.randn(b_size, z_dim, latent_size, latent_size, device=device)
        G_outputs = G(noise) #fake images

        output = D(G_outputs.detach()).view(-1)
        D_loss_fake = criterion(output, torch.zeros_like(output))
        D_G_z1 = output.mean().item()

        # Compute Error, backpropagate, optimize
        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        optimizerD.step()

        # --- Train Generator ---
        G.zero_grad()
        # Forward G into D and calculate loss
        output = D(G_outputs).view(-1)
        G_loss = criterion(output, torch.ones_like(output))

        # Backpropagate, store z2 and optimize
        G_loss.backward()
        D_G_z2 = output.mean().item() # should slightly change (D has been optimized)
        optimizerG.step()

        # Training stats
        if i % 20 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2), flush=True)
            
        # Save loss
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())
