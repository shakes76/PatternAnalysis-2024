import numpy as np 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset import get_dataloaders
from modules import *

batch_size = 32
n_updates = 5000
n_hiddens = 128
n_residual_hiddens = 32
n_residual_layers = 2
embedding_dim = 64
n_embeddings = 512
beta = 0.25
learning_rate = 3e-4
log_interval = 50
n_epochs = 15

def train():

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

    for epoch in range(n_epochs):
        # (x, _) = next(iter(train_loader))
        for im in tqdm(train_loader):
            im = im.float().unsqueeze(1).to(device)
            print(im.shape)
            # x = next(iter(train_loader))
            # x = x.to(device)
            optimiser.zero_grad()

            embedding_loss, x_hat, perplexity = model(im)
            recon_loss = criterion(x_hat, im)
            # recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimiser.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = epoch

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
                
        
        print(f"Finished epoch {epoch + 1}")
    
    print("---Training Complete---")

if __name__ == "__main__":
    train()