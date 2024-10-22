"""
This file contains the source code for training, validating, testing and saving the model.

The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.

The losses and other metric will be recorded and plotted here.
"""
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import create_nifti_data_loaders
from modules import VQVAE2
import numpy as np


def train_epoch(model, data_loader, optimiser, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The VQVAE2 model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimiser (torch.optim.Optimizer): The optimizer used to update model parameters.
        device (torch.device): The device (CPU or GPU) used for training.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        batch = batch.to(device)

        optimiser.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate_epoch(model, data_loader, device):
    """
    Validates the model for one epoch.

    Args:
        model (torch.nn.Module): The VQVAE2 model to validate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the validation data.
        device (torch.device): The device (CPU or GPU) used for validation.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            loss, _ = model(batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def plot_losses(train_losses, val_losses, output_dir):
    """
    Plots training and validation losses and saves the plot to the specified directory.

    Args:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
        output_dir (str): Directory to save the plot.
    """
    plt.figure()
    plt.plot(train_losses, label = 'Training Loss')
    plt.plot(val_losses, label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()


def save_model(model, epoch, output_dir):
    """
    Saves the model's state dictionary to the specified directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        epoch (int or str): The current epoch or 'final' to denote the final model save.
        output_dir (str): Directory to save the model.
    """
    model_path = os.path.join(output_dir, f'vqvae2_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')


def main(
        data_dir,
        output_dir,
        batch_size = 16,
        num_epochs = 10,
        lr = 1e-3,
        hidden_dims = [64, 128],
        num_embeddings = [256, 256],
        embedding_dims = [32, 64],
        commitment_cost = 0.25,
        num_workers = 4):
    """
    Main function to train and validate the VQVAE2 model on MRI data.

    Args:
        data_dir (str): Directory containing training and validation data.
        output_dir (str): Directory to save the trained model and loss plots.
        batch_size (int): Batch size for data loading. Default is 16.
        num_epochs (int): Number of training epochs. Default is 10.
        lr (float): Learning rate for the optimizer. Default is 1e-3.
        hidden_dims (list): List of hidden dimensions for the model layers.
        num_embeddings (list): List of codebook sizes for each level of VQVAE2.
        embedding_dims (list): List of embedding dimensions for each level of VQVAE2.
        commitment_cost (float): Weight for the commitment loss in VQVAE2.
        num_workers (int): Number of worker threads for data loading. Default is 4.
    """
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used:", device, "\n")

    # Data loaders
    print("Starting data loading.")
    train_loader = create_nifti_data_loaders(os.path.join(data_dir, 'keras_slices_train'), batch_size, num_workers)
    val_loader = create_nifti_data_loaders(os.path.join(data_dir, 'keras_slices_validate'), batch_size, num_workers)
    print("Done loading data.\n")

    # Initialize the model
    in_channels = 1
    model = VQVAE2(in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost).to(device)

    # Check the number of parameters in the model
    print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))

    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr = lr)

    # Lists to store losses
    train_losses = []
    val_losses = []

    # Training loop
    print("\nStarting Training:")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimiser, device)
        train_losses.append(train_loss)

        # Validate for one epoch
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)

        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the model periodically
        save_model(model, 'perodical', output_dir)

        # Save the train and validation losses using numpy
        np.savetxt(os.path.join(output_dir, 'train_losses_periodical.txt'), np.array(train_losses))
        np.savetxt(os.path.join(output_dir, 'val_losses_periodical.txt'), np.array(val_losses))

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses, output_dir)

    # Save the train and validation losses using numpy
    np.savetxt(os.path.join(output_dir, 'train_losses.txt'), np.array(train_losses))
    np.savetxt(os.path.join(output_dir, 'val_losses.txt'), np.array(val_losses))

    # Save final model
    save_model(model, 'final', output_dir)


if __name__ == "__main__":
    current_location = os.getcwd()
    data_dir = os.path.join(current_location, 'recognition', 'VQVAE_s4803279', 'HipMRI_study_keras_slices_data')
    output_dir = 'trained_vqvae2_model'
    main(data_dir, output_dir)
