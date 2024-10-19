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


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            loss, _ = model(batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def plot_losses(train_losses, val_losses, output_dir):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()


def save_model(model, epoch, output_dir):
    model_path = os.path.join(output_dir, f'vqvae2_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')


def main(
        data_dir, 
        output_dir, 
        batch_size = 16, 
        num_epochs = 1, 
        lr = 1e-3, 
        hidden_dims = [128, 256], 
        num_embeddings = [512, 1024], 
        embedding_dims = [64, 64], 
        commitment_cost = 0.25, 
        num_workers = 4):
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader = create_nifti_data_loaders(os.path.join(data_dir, 'keras_slices_train'), batch_size, num_workers)
    val_loader = create_nifti_data_loaders(os.path.join(data_dir, 'keras_slices_validate'), batch_size, num_workers)

    # Initialize the model
    in_channels = 1  # Assuming single-channel (grayscale) images
    model = VQVAE2(in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to store losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1)):
        print(f'Epoch {epoch}/{num_epochs}')

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate for one epoch
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)

        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # # Save the model periodically
        # if epoch % 10 == 0:
        #     save_model(model, epoch, output_dir)

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses, output_dir)

    # # Save final model
    # save_model(model, 'final', output_dir)


if __name__ == "__main__":
    data_dir = r"C:\Users\GamingPC\OneDrive - The University of Queensland\School\2024\Sem2\COMP3710\Project\HipMRI_study_keras_slices_data"
    output_dir = './trained_vqvae2_model'
    main(data_dir, output_dir)
