import glob
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import NiftiDataset
from modules import Encoder, Decoder, VectorQuantizer, VQVAE
import numpy as np
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




num_epochs = 10
batch_size = 16
learning_rate = 1e-4
num_emb = 512
e_dim = 64
commit_cost = 0.25
n_res_layers = 2
res_h_dim = 32
h_dim = 64


input_transf = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.Normalize((0.5,), (0.5,))
    ])

hipmri_train_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_train"
hipmri_val_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_validate"
hipmri_test_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_test"

base_dir = os.getcwd()
hipmri_train_dir = os.path.join(base_dir, hipmri_train_dir)
hipmri_val_dir = os.path.join(base_dir, hipmri_val_dir)
hipmri_test_dir = os.path.join(base_dir, hipmri_test_dir)
#hipmri_train_dir = "keras_slices_data/keras_slices_train"
#hipmri_valid_dir= "keras_slices_data/keras_slices_validate"
#hipmri_test_dir= "keras_slices_data/keras_slices_test"
save_path = ""


training_set, validation_set, test_set = NiftiDataset.get_dataloaders(train_dir=hipmri_train_dir, val_dir=hipmri_val_dir, test_dir=hipmri_test_dir, batch_size=16, num_workers=4, transform=input_transf)

model = VQVAE(h_dim, res_h_dim, n_res_layers, num_emb, e_dim, commit_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(recon_loss, img, embed_loss):
    bce = F.binary_cross_entropy(recon_loss, img)
    total_loss = bce + embed_loss
    return total_loss

# Training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in training_set:
        optimizer.zero_grad()
        batch = batch.to(device)
        scalar_loss, x_hat, scalar_metric = model(batch)
        # print(f"OUTPUT OF MODEL(BATCH): {x_hat}")

        loss = 1 - ssim(batch, x_hat, data_range=1, size_average=True)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(training_set)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

