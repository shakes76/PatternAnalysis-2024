import numpy as np 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from torch.optim import Adam
from dataset import get_dataloaders
from modules import *
from utils import *
import matplotlib.pyplot as plt

batch_size = 32
n_updates = 5000
n_hiddens = 128
n_residual_hiddens = 32
n_residual_layers = 2
embedding_dim = 64
n_embeddings = 512
beta = 0.25
learning_rate = 3e-4
log_interval = 5
n_epochs = 15

# Model setup"
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
train_loader, test_loader = get_dataloaders(batch_size)
model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers, n_embeddings, embedding_dim, beta).to(device)

optimiser = Adam(model.parameters(), lr=learning_rate, amsgrad=True)
criterion = torch.nn.MSELoss()

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

losses = []
ssim_scores = []
best_ssim_epoch = 0

def train():
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

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = epoch

            epoch_losses.append(loss.cpu().detach().numpy())

            if epoch % log_interval == 0:
                # """
                # save model and print values
                # """
                # if save:
                #     hyperparameters = __dict__
                #     utils.save_model_and_results(
                #         model, results, hyperparameters, filename)

                print('Update #', epoch, 'Recon Error:',
                    np.mean(results["recon_errors"][-log_interval:]),
                    'Loss', np.mean(results["loss_vals"][-log_interval:]),
                    'Perplexity:', np.mean(results["perplexities"][-log_interval:]))
        
        losses.append(np.mean(epoch_losses))
        epoch_ssim(epoch)
        model.train()
                
        
        print(f"Finished epoch {epoch + 1}")
    
    print("---Training Complete---")

def epoch_ssim(epoch):
    model.eval()
    total_ssim = 0

    with torch.no_grad():
        for batch_no, x in enumerate(test_loader):
            # x = x.to(device)
            x = x.float().unsqueeze(1).to(device)
            _, x_hat, _ = model(x)
            total_ssim += batch_ssim(x_hat, x)
    
    current_ssim = total_ssim / (batch_no + 1)
    ssim_scores.append(current_ssim)

    if current_ssim == max(ssim_scores):
        best_ssim_epoch = epoch
        print(f"New best epoch: {epoch} with {current_ssim}") 


def plot_epochs(y, name):
    print("---Graphing Losses---")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), y, label=name)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    plt.grid()
    plt.savefig(f"{name}.png")
    plt.close()


if __name__ == "__main__":
    print(f"Time: {time.time()}")
    train()
    plot_epochs(losses, "losses")
    plot_epochs(ssim_scores, "ssim_scores")
    print("done")