"""
Contains the VQ-VAE training loop with image generation and metric plotting.

Author: Kirra Fletcher s4745168
"""
import numpy as np 
import os
import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from dataset import *
from modules import *
from utils import *
from predict import *

# Directories for Nifti files:
train_file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
test_file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

# Default Hyperparameters:
n_hiddens = 128
n_residual_hiddens = 32
embedding_dim = 64
n_embeddings = 512
beta = 0.25
learning_rate = 3e-4

# Hyperparameters changed from default:
batch_size = 8
n_epochs = 150
n_residual_layers = 5

# Set up directory to save Images and Plots
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
directory = f'./Project/{current_time}'
if not os.path.exists(directory):
    os.makedirs(directory)
else:
    os.makedirs(f'{directory}_{datetime.now().microsecond}')

# Model setup:
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
train_loader, test_loader, val_loader = get_dataloaders(batch_size, train_file_path, test_file_path)
model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers, n_embeddings, embedding_dim, beta).to(device)

optimiser = Adam(model.parameters(), lr=learning_rate, amsgrad=True)
criterion = torch.nn.MSELoss()

model.train()

# Record Results:
losses = []
ssim_scores = []
best_ssim_epoch = 0

def train():
    """
    Trains the VQ-VAE model for `n_epochs` epochs, using the training data. Each epoch the model is
    tested using the validation data, using the SSIM score. The average log(loss) of an epoch is recorded.
    """
    for epoch in range(n_epochs):
        epoch_losses = []

        for im in tqdm(train_loader):
            im = im.float().unsqueeze(1).to(device)
            optimiser.zero_grad()

            embedding_loss, x_hat, perplexity = model(im)
            recon_loss = criterion(x_hat, im)
            loss = recon_loss + embedding_loss

            loss.backward()
            optimiser.step()

            epoch_losses.append(math.log(loss.cpu().detach().numpy()))

        losses.append(np.mean(epoch_losses))
        epoch_ssim(epoch)
        
        if epoch > 0:
            plot_loss(losses, directory, epoch)
            plot_ssim(ssim_scores, directory, epoch)

        model.train()    
        
        
def epoch_ssim(epoch):
    """
    Calculates the SSIM score of the model using the validation dataset.

    Parameters:
        epoch: Current epoch at which the model is being evaluated.
    """
    model.eval()  # Eval mode to not train on the validation set

    batch_ssims = []
    with torch.no_grad():
        for x in val_loader:
            x = x.float().unsqueeze(1).to(device)
            _, x_hat, _ = model(x)
            batch_ssims.append(batch_ssim(x_hat, x))
    
    average_ssim = np.mean(batch_ssims)
    ssim_scores.append(average_ssim)
    if average_ssim == max(ssim_scores):  # New best model! Save the model and generate images
        best_ssim_epoch = epoch
        generate(best_ssim_epoch, model, val_loader, directory, average_ssim)
        torch.save(model.state_dict(), f'{directory}/best_model.pt')
        print(f"New best epoch: {best_ssim_epoch} with {average_ssim}") 


def final_ssim():
    """
    Calculates the SSIM score of the final model - i.e. the best model after all epochs are complete. 
    Final SSIM score is calculated using the test data set (to ensure model generalises well).
    """
    model.load_state_dict(torch.load(f'{directory}/best_model.pt')) 
    model.eval()

    batch_ssims = []
    with torch.no_grad():
        for x in test_loader:
            x = x.float().unsqueeze(1).to(device)
            _, x_hat, _ = model(x)
            batch_ssims.append(batch_ssim(x_hat, x))
    
    final_ssim = np.mean(batch_ssims)
    return final_ssim


def plot_loss(losses, directory, epoch):
    """
    Plots the log(loss) of the model training. 

    Parameters:
        losses: log(loss) of each epoch (on y-axis)
        directory: Directory to save the plot in
        epoch: Current epoch that is being plotted (Allows for continuous update of loss through long training)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epoch + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel("log(Loss)")
    plt.grid()
    plt.savefig(f"{directory}/losses.png")
    plt.close()

def plot_ssim(ssim_scores, directory, epoch):
    """
    Plots the SSIM scores of the model up to the current epoch.

    Parameters:
        ssim_scores: SSIM of each epoch (on y-axis)
        directory: Directory to save the plot in
        epoch: Current epoch that is being plotted (Allows for continuous update of loss through long training)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epoch + 1), ssim_scores)
    plt.xlabel('Epoch')
    plt.ylabel("SSIM")
    plt.grid()
    plt.savefig(f"{directory}/ssim_scores.png")
    plt.close()


if __name__ == "__main__":
    """
    Main function which completes model training, plotting of results, and generation of images.
    """
    start = time.time()
    print(f"Time: {current_time}")
    train()
    print(f"TRAINING COMPLETE - took {(time.time() - start) // 60} minutes")
    plot_loss(losses, directory, n_epochs - 1)
    plot_ssim(ssim_scores, directory, n_epochs - 1)
    test_ssim = final_ssim()
    generate(best_ssim_epoch, model, test_loader, directory, test_ssim)
    