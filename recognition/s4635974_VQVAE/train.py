"""
COMP3710 VQ-VAE report
Author: David Collie
Date: October, 2024

VQ-VAE Training Script

This script defines and trains a Vector Quantized Variational Autoencoder (VQ-VAE) model on the HIP Mri dataset for prostate cancer. It includes the following key components:

1. **ValidationLossEarlyStopping Class**: A helper class that monitors the validation loss during training and implements early stopping functionality to prevent overfitting. The training will halt if the validation loss does not improve for a specified number of epochs.

2. **Hyperparameters**: The script sets hyperparameters for training, including the number of epochs, batch size, learning rate, model architecture parameters, and the use of early stopping.

3. **Train Model Function**: The `train_model` function orchestrates the training process, including:
   - Loading the MRI datasets using the `HipMRILoader` class.
   - Initializing the VQ-VAE model and the optimizer.
   - Executing the training loop where the model learns from the training dataset.
   - Evaluating the model on a validation dataset after each epoch, calculating reconstruction loss, VQ loss, and Structural Similarity Index Measure (SSIM) for quality assessment.
   - Saving images of real and decoded outputs every 10 epochs for visual inspection of the model's performance.
   - Implementing early stopping based on validation loss if specified.

4. **Data Visualization**: At the end of training, the script saves training and validation loss data for visualization, enabling the creation of plots to analyze the model's performance over epochs.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.image \
    import structural_similarity_index_measure as ssim
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from dataset import HipMRILoader
import modules

class ValidationLossEarlyStopping:
    """Class to handle early stopping based on validation loss.

    Attributes:
        patience (int): Number of epochs to wait for an improvement in validation loss before stopping.
        min_delta (float): Minimum change in validation loss to qualify as an improvement.
        counter (int): Counts how many consecutive epochs with no significant improvement.
        min_validation_loss (float): The best validation loss seen so far.
    """
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
        """Checks if training should be stopped based on validation loss.

        Args:
            validation_loss (float): The validation loss of the current epoch.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if ((validation_loss + self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False

# Hyperparmeters for loading model
num_epochs = 150           # Number of training epochs
batch_size = 16            # Batch size for the dataloader
lr = 0.0008                 # Learning rate for the optimizer
num_hiddens = 128          # Number of hidden units in the VQ-VAE model
num_residual_hiddens = 32  # Number of residual hidden units
num_channels = 1           # Number of input channels (grayscale image)
num_embeddings = 512       # Number of embeddings in the VQ-VAE model
dim_embedding = 64         # Dimensionality of each embedding
beta = 0.25                # Beta parameter for commitment loss in VQ-VAE
early_stopping = True

# Enter a description of model. Used for identifying saved files
model_description = 'VQ-VAE'

# Directory for saving images is named after model description 
save_dir = model_description

# Model is saved at the end of training in saved_model folder using description of model
model_dir = f'saved_model/{model_description}.pth'

# Data of model is saved for making plots. 
save_data_dir = f'data_viz/{model_description}.pkl'

def train_model(
        save_dir: str | None = None, 
        model_dir: str | None = None, 
        save_data_dir: str | None = None,
        num_epochs: int = num_epochs,
        batch_size: int = batch_size,
        lr: float = lr,
        num_hiddens: int = num_hiddens,
        num_residual_hiddens: int = num_residual_hiddens,
        num_channels: int = num_channels,
        num_embeddings: int = num_embeddings,
        dim_embedding: int = dim_embedding,
        beta: float = beta,
        early_stopping: bool = early_stopping
        ):
    """Main function to train the VQ-VAE model.

    Args:
        save_dir (str): Directory for saving results and images.
        model_dir (str): Directory for saving the model.
        save_data_dir (str): Directory for saving data for plotting.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        num_hiddens (int): Number of hidden units.
        num_residual_hiddens (int): Number of residual hidden units.
        num_channels (int): Number of channels in the input data.
        num_embeddings (int): Number of embeddings in the VQ-VAE.
        dim_embedding (int): Dimension of each embedding.
        beta (float): Beta parameter for VQ-VAE loss.
        early_stopping (bool): Flag for using early stopping.
    """

    # Printing hyperparameters so that each training script has a record
    print("==== Paramaters ====")
    print(f"Max Epochs: ", num_epochs)
    print(f"Batch size: ", batch_size)
    print(f"lr: ", lr)
    print(f"num_hiddens: ", num_hiddens)
    print(f"num_residual_hiddens: ", num_residual_hiddens)
    print(f"num_channels", num_channels)
    print(f"num_embeddings: ", num_embeddings)
    print(f"dim_embedding: ", dim_embedding)
    print(f"beta: ", beta)
    print()

    # Save directory
    if type(save_dir) == str:
        os.makedirs(save_dir, exist_ok=True)

    # Model directory
    if type(model_dir) == str:
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    # Data vizulisation directory
    if type(save_data_dir) == str:
        os.makedirs(os.path.dirname(save_data_dir), exist_ok=True)

    # Configure Pytorch
    seed = 42
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    print()

    # Directories for datasets
    train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
    test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
    validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

    # Get loaders
    train_loader, validate_loader, data_variance = HipMRILoader(
        train_dir, validate_dir, test_dir,
        batch_size=batch_size, transform=None
        ).get_loaders()

    # Create model
    model = modules.VQVAE(
        num_channels=num_channels,
        num_hiddens=num_hiddens,
        num_residual_hiddens=num_hiddens,
        num_embeddings=num_embeddings,
        dim_embedding=dim_embedding,
        beta=beta).to(device)

    # Set optimiser
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        amsgrad=False)

    # Initiate early stopping, if used
    if early_stopping:
        early_stopper = ValidationLossEarlyStopping(40, 0.01)

    # Training mectrics
    epoch_training_reconstruction_loss = []
    epoch_training_vq_loss = []

    epoch_validation_reconstruction_loss = []
    epoch_validation_vq_loss = []
    epoch_ssim = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        training_reconstruction_error = []
        training_vq_error = []

        # Training set loop
        for _, training_images in enumerate(train_loader):
            training_input_images = training_images.to(device)

            optimizer.zero_grad()
            vq_loss, training_output_images = model(training_input_images)

            # Calculate reconstruction loss
            reconstruction_loss = F.mse_loss(training_output_images, training_input_images)
            loss = reconstruction_loss + vq_loss
            loss.backward()

            optimizer.step()

            training_reconstruction_error.append(reconstruction_loss.item())
            training_vq_error.append(vq_loss.item())
        
        # Calculate and store average training losses
        training_reconstruction_loss = np.mean(training_reconstruction_error)
        epoch_training_reconstruction_loss.append( training_reconstruction_loss)
        training_vq_loss = np.mean(training_vq_error)
        epoch_training_vq_loss.append(training_vq_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training reconstruction loss: { training_reconstruction_loss:.5f}, Training VQ loss: {training_vq_loss}')

        # Evaluate on the validation dataset
        model.eval()
        validation_reconstruction_error = []
        validation_vq_error = []
        validation_ssim = []
        
        # Validation set loop
        with torch.no_grad():
            for j, validation_images in enumerate(validate_loader):
                validation_input_images = validation_images.to(device)
                validation_vq_loss, validation_output_images = model(validation_input_images)

                # Reshape images for SSIM calculation
                real_image = validation_input_images.view(-1, 1, 256, 128).detach().to(device)
                decoded_image = validation_output_images.view(-1, 1, 256, 128).detach().to(device)
                
                # Calculate SSIM and store it
                similarity = ssim(decoded_image, real_image).item()
                validation_ssim.append(similarity)

                # Calculate reconstruction loss for validation
                validation_reconstruction_loss = F.mse_loss(validation_output_images, validation_input_images)
                validation_reconstruction_error.append(validation_reconstruction_loss.item())
                validation_vq_error.append(validation_vq_loss.item())

        # Average validation loss and SSIM
        average_validation_loss = np.mean(validation_reconstruction_error)
        epoch_validation_reconstruction_loss.append(average_validation_loss)
        average_validation_vq_loss = np.mean(validation_vq_error)
        epoch_validation_vq_loss.append(average_validation_vq_loss)
        average_ssim = np.mean(validation_ssim)
        epoch_ssim.append(average_ssim)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation reconstruction loss: {average_validation_loss:.5f}, Validation VQ loss: {average_validation_vq_loss}, Average SSIM: {average_ssim:.5f}')

        # Save resl vs. decoded image after every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Number of images to save
            num_images_to_save = 4  # Adjust this for the number of real and decoded images

            # Convert tensors to numpy arrays
            real_images = validation_input_images.cpu().numpy()
            decoded_images = validation_output_images.cpu().numpy()  

            # Create a new figure for saving real and decoded images side by side
            fig, axes = plt.subplots(num_images_to_save, 2, figsize=(8, num_images_to_save * 4))  # 2 columns

            # Plot real and decoded images
            for k in range(num_images_to_save):
                # Plot real images
                axes[k, 0].imshow(real_images[k, 0], cmap='gray')
                axes[k, 0].set_title('Real Image')
                axes[k, 0].axis('off')

                # Plot decoded images
                axes[k, 1].imshow(decoded_images[k, 0], cmap='gray')
                axes[k, 1].set_title('Decoded Image')
                axes[k, 1].axis('off')

            # Add a title for the entire figure with the epoch number
            fig.suptitle(f'Epoch {epoch + 1}', fontsize=16)

            # Adjust layout for better spacing
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title

            # Save the figure as a single PNG file with the epoch number in the filename
            plt.savefig(os.path.join(save_dir, f'real_and_decoded_images_epoch_{epoch + 1}.png'))
            plt.close()

        # Early stopping, if used
        if early_stopping:
            # Check if early stopping condition is reached
            if early_stopper.early_stop_check(average_validation_loss):
                print(f"Stopped early at Epoch: {epoch +1}")
                break
        
        print()

    # After training, save the lists to a file
    data = {
        "training_reconstruction_loss": epoch_training_reconstruction_loss,
        "training_vq_loss": epoch_training_vq_loss,
        "validation_reconstruction_loss": epoch_validation_reconstruction_loss,
        "validation_vq_loss": epoch_validation_vq_loss,
        "ssim": epoch_ssim
    }

    if type(save_data_dir) == str:
        with open(save_data_dir, 'wb') as f:
            pickle.dump(data, f)

    if type(save_dir) == str:
        epochs = range(1, len(epoch_training_reconstruction_loss) + 1)

        # 1. Plot Training reconstruction Loss vs. Validation reconstruction Loss
        plt.figure(figsize=(12, 6))  # Width of normal page
        plt.plot(epochs, epoch_training_reconstruction_loss, label='Training Reconstruction Loss')
        plt.plot(epochs, epoch_validation_reconstruction_loss, label='Validation Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('reconstruction Loss')
        plt.title('Training and Validation Reconstruction Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'reconstruction_loss_per_epoch.png'))
        plt.close()

        # 2. Plot Training VQ Loss vs. Validation VQ Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, epoch_training_vq_loss, label='Training VQ Loss')
        plt.plot(epochs, epoch_validation_vq_loss, label='Validation VQ Loss')
        plt.xlabel('Epochs')
        plt.ylabel('VQ Loss')
        plt.title('Training and Validation VQ Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'vq_loss_per_epoch.png'))
        plt.close()

        # 3. Plot Combined Training Loss vs. Validation Loss
        combined_training_loss = [x + y for x, y in zip(epoch_training_reconstruction_loss, epoch_training_vq_loss)]
        combined_validation_loss = [x + y for x, y in zip(epoch_validation_reconstruction_loss, epoch_validation_vq_loss)]

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, combined_training_loss, label='Combined Training Loss')
        plt.plot(epochs, combined_validation_loss, label='Combined Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Combined Loss')
        plt.title('Combined Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'combined_loss_per_epoch.png'))
        plt.close()

        # 4. Plot SSIM per Epoch
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, epoch_ssim, label='SSIM')
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.title('SSIM per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'ssim_per_epoch.png'))
        plt.close()

    if type(model_dir) == str:
        # Save the model state_dict
        torch.save(model.state_dict(), model_dir)

    print("End")

    return model

if (__name__ == "__main__"):
    model = train_model(
        save_dir=save_dir, 
        model_dir=model_dir,
        save_data_dir=save_data_dir)

