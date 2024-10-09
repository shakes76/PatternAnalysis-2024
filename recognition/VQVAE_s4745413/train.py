import glob
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import NiftiDataset
from modules import Encoder, Decoder, VectorQuantizer, VQVAE
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




num_epoch = 10
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

hipmri_train_dir = "../../../keras_slices_data/keras_slices_train"
hipmri_valid_dir= "../../../keras_slices_data/keras_slices_validate"
hipmri_test_dir= "../../../keras_slices_data/keras_slices_test"
save_path = ""


training_set, validation_set, test_set = NiftiDataset.get_dataloaders(train_dir=hipmri_train_dir, val_dir=hipmri_valid_dir, test_dir=hipmri_test_dir)

model = VQVAE(h_dim, res_h_dim, n_res_layers, num_emb, e_dim, commit_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
