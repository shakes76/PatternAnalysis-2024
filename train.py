import torch
import torch.optim as optim
import argparse
import utils
import dataset
from modules import VQVAE
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
epochs = 100
learning_rate = 1e-3
batch_size = 16
weight_decay = 1e-5

# define model architecture
#
n_hiddens = 512
n_residual_hiddens = 256
n_residual_layers = 16
embedding_dim = 512
n_embeddings = 1024
beta = 0.1

# Add dataset and model save arguments for easier use and storage
parser.add_argument("--dataset_dir", type=str, default='HipMRI_study_keras_slices_data')  
parser.add_argument("--norm_image", type=bool, default=False)
parser.add_argument("--categorical", type=bool, default=False)
parser.add_argument("-save", action="store_true")

args = parser.parse_args() # argument parser 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

if args.save: # specific argument to save the model so training can be run without affecting current saved model, testing purposes
    print('Results will be saved in ./results/vqvae_data.pth')

"""
Load data and define batch data loaders for .nii files
"""
# locate file location of training a validate data
train_path = os.path.join(args.dataset_dir, 'keras_slices_train')
validate_path = os.path.join(args.dataset_dir, 'keras_slices_validate')

# retrieve all .nii files from the specified data location
nii_files_train = [os.path.join(train_path, img) for img in os.listdir(train_path) if img.endswith(('.nii', '.nii.gz'))]
nii_files_validate = [os.path.join(validate_path, img) for img in os.listdir(validate_path) if img.endswith(('.nii', '.nii.gz'))]

# extract data from the .nii files
x_train = dataset.load_data_2D(nii_files_train, normImage=args.norm_image, categorical=args.categorical)
x_val = dataset.load_data_2D(nii_files_validate, normImage=args.norm_image, categorical=args.categorical)

# convert data to tensors for use in torch, plus adding channel dimension
x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1) 
x_val_tensor = torch.from_numpy(x_val).float().unsqueeze(1)

# create dataloader for use in training
train_loader = torch.utils.data.DataLoader(x_train_tensor, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(x_val_tensor, batch_size=batch_size)

# initialise model
model = VQVAE(n_hiddens, n_residual_hiddens,
              n_residual_layers, n_embeddings, embedding_dim, beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

model.train()

# results dictionary to store model data
results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

def train():

    for epoch in range(epochs): 
        # use of tqdm to create a progress bar of training batches for each epoch
        for i, (x) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')):

            x = x.to(device) # move batch to device (preferably gpu)
            optimizer.zero_grad() # zero out gradients

            # forward pass
            embedding_loss, x_hat, perplexity = model(x)

            # compute reconstruction loss
            recon_loss = torch.mean((x_hat - x)**2)
            loss = recon_loss + embedding_loss

            # backwards pass and optimization
            loss.backward()
            optimizer.step()

            # store results for logging and saving
            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = i

        # save the model and data
        if args.save:
            hyperparameters = args.__dict__
            utils.save_model_and_results(
                model, results, hyperparameters)

if __name__ == "__main__":
    train()
