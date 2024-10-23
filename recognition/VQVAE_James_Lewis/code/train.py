import torch
import os
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from modules import VQVAE
from dataset import load_data_2D, DataLoader
import torchmetrics.image
import torch.optim as optim
import matplotlib.pyplot as plt
import torchmetrics


def train_vqvae(model, train_images, val_images, test_images, num_epochs, learning_rate, device, batch_size,
                lambda_ssim=0.1, max_grad_norm=1.0):
    model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Initialize SSIM metric
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False)

    ssim_vals = []
    ssim_vals_valid = []
    # Training loop
    save_original(train_loader)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # Track total loss for the epoch
        total_ssim = 0  # Track total SSIM for the epoch

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
            # Convert to tensor and ensure it's in the correct shape
            data = data.unsqueeze(1)
            # Add a channel dimension for grayscale [batch_size, 1, height, width]
            data = data.to(device)  # Move to device

            # Forward pass through the model
            reconstructed_data, commitment_loss, embeddings = model(data)


            # Calculate reconstruction loss (Mean Squared Error)
            reconstruction_loss = criterion(reconstructed_data, data)

            # Compute SSIM between the reconstructed images and original images
            ssim_score = ssim_metric(reconstructed_data, data)
            ssim_vals.append(ssim_score.item())

            total_loss_value = reconstruction_loss + commitment_loss * commitment_cost
            total_loss_value.backward()  # Backpropagate the loss

            # Gradient Clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update the weights
            optimizer.step()

            # Accumulate total SSIM and total loss for reporting
            total_loss += total_loss_value.item()
            total_ssim += ssim_score.item()

            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_loss_value.item():.4f}, '
                      f'Reconstruction Loss (MSE): {reconstruction_loss.item():.4f}, '
                      f'SSIM: {ssim_score.item():.4f}')

        # Step the learning rate scheduler
        scheduler.step()

        # Average loss and SSIM for the epoch
        avg_loss = total_loss / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average SSIM: {avg_ssim:.4f}')

        # Validate the model on the validation set
        ssim_valid = validate(model, val_loader, device, ssim_metric, ssim_vals_valid)
        ssim_vals_valid.append(ssim_valid)

        construct_images(data, reconstructed_data, embeddings, epoch, section = 'train')

    return ssim_vals, ssim_vals_valid

def validate(model, val_loader, device, ssim_metric, ssim_vals_valid):
    """
    Validate the model on validation data.

    @param model: VQVAE, the model being validated
    @param val_loader: DataLoader, validation data
    @param device: torch.device, the device for computations
    @param ssim_metric: torchmetrics.Metric, SSIM metric instance
    """
    model.eval()  # Set the model to evaluation mode
    total_val_ssim = 0  # Track total SSIM for validation


    with torch.no_grad():
        for data in val_loader:
            # Convert the numpy array to a PyTorch tensor
            data = data.unsqueeze(1)
            # Add a channel dimension for grayscale [batch_size, 1, height, width]
            data = data.to(device) # Move the data to the appropriate device (e.g., GPU)

            # Forward pass through the model
            reconstructed_data, _ = model(data)

            # Compute SSIM between reconstructed and original data
            ssim_score = ssim_metric(reconstructed_data, data)
            # Accumulate SSIM score
            total_val_ssim += ssim_score.item()

    # Return average SSIM over validation set
    avg_val_ssim = total_val_ssim / len(val_loader)
    print(f'Validation SSIM: {avg_val_ssim:.4f}')
    return ssim_score

def test_model(model, test_loader, device, ssim_metric):
    """
    Test the model on the test set.

    @param model: VQVAE, the model being tested
    @param test_loader: DataLoader, test data
    @param device: torch.device, the device for computations
    @param ssim_metric: torchmetrics.Metric, SSIM metric instance
    """
    model.eval()  # Set the model to evaluation mode
    total_test_ssim = 0  # Track total SSIM for test

    with torch.no_grad():
        for data in test_loader:
            # Convert the numpy array to a PyTorch tensor
            data = data.unsqueeze(1)  # Add channel dimension for grayscale [batch_size, 1, height, width]
            data = data.to(device)  # Move the data to the appropriate device (e.g., GPU)

            # Forward pass through the model
            reconstructed_data, _ = model(data)

            # Compute SSIM between reconstructed and original data
            ssim_score = ssim_metric(reconstructed_data, data)

            # Accumulate SSIM score
            total_test_ssim += ssim_score.item()

    # Return average SSIM over test set
    avg_test_ssim = total_test_ssim / len(test_loader)
    print(f'Test SSIM: {avg_test_ssim:.4f}')


def plot_ssims(ssim_vals_train, ssim_vals_val, num_epochs, save_path='average_ssim_over_epochs.png'):
    """
    Plot the SSIM values over training epochs for both training and validation.

    @param ssim_vals_train: list, SSIM values for training
    @param ssim_vals_val: list, SSIM values for validation
    @param num_epochs: int, total number of epochs
    @param save_path: str, optional, path to save the plot image
    """

    plt.figure(figsize=(10, 5))

    # Plot training SSIM values (line-only)
    plt.plot(range(1, num_epochs + 1), ssim_vals_train, label='Training SSIM', color='blue', linewidth=2)

    # Plot validation SSIM values (line-only)
    plt.plot(range(1, num_epochs + 1), ssim_vals_val, marker='x', label='Validation SSIM', color='orange')

    # Add titles and labels
    plt.title('Average SSIM over Epochs (Training vs Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Average SSIM')
    plt.grid()

    # Ensure all epochs are shown on x-axis
    plt.xticks(range(1, num_epochs + 1))

    # Add a legend to distinguish training and validation SSIM values
    plt.legend()

    # Save the figure to a file
    plt.savefig(save_path)
    print(f'Plot saved to {save_path}')

    # Display the plot
    plt.show()
def plot_losses(losses, num_epochs, save_path='loss_over_epochs.png'):
    """
    Plot the losses over training epochs.

    @param losses: list, loss values over epochs
    @param num_epochs: int, total number of epochs
    @param save_path: str, optional, path to save the plot image
    """

    plt.figure(figsize=(10, 5))

    # Plot the losses (line-only)
    plt.plot(range(1, num_epochs + 1), losses, label='Loss', color='blue', linewidth=2)

    # Add titles and labels
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    # Ensure all epochs are shown on x-axis
    plt.xticks(range(1, num_epochs + 1))

    # Save the figure to a file
    plt.savefig(save_path)
    print(f'Plot saved to {save_path}')

    # Display the plot
    plt.show()

def save_original(train_loader):
    original_imgs = next(iter(train_loader))
    original_imgs = (original_imgs*0.5) + 0.5

    grid =make_grid(original_imgs, nrow=8)

    os.makedirs('original_images', exist_ok=True)
    save_image(grid, 'original_images/original_images.png')


    return
def construct_images(original_imag, reconstructed_data, embeddings, epoch, section):
    """
    Construct and save the original and reconstructed images for visualization.

    @param data: torch.Tensor, the original images
    @param reconstructed_data: torch.Tensor, the reconstructed images
    @param embeddings: torch.Tensor, the embeddings
    @param epoch: int, the current epoch
    @param section: str, the section of the dataset (e.g., 'train', 'val', 'test')
    """
    save_dir = f'reconstructed_images/{section}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    original_imag = original_imag.detach().cpu()
    reconstructed_data = reconstructed_data.detach().cpu()
    embeddings = embeddings.detach().cpu()

    num_images = min(5, original_imag.size(0))  # Limit to first 5 images for clarity
    fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(num_images * 3, 9))

    for i in range(num_images):
        # Original Image
        axes[0, i].imshow(original_imag[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i + 1}')

        # Reconstructed Image
        axes[1, i].imshow(reconstructed_data[i].squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Reconstructed {i + 1}')

        # Embedding Visual (flatten embeddings to visualize if necessary)
        embedding_img = embeddings[i].view(8, 8).numpy()  # Assuming embedding size is 64 for visualization
        axes[2, i].imshow(embedding_img, cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Embedding {i + 1}')

    plt.suptitle(f'{section.capitalize()} Images - Epoch {epoch}')

    # Save the image grid to a file
    save_path = os.path.join(save_dir, f'epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

    print(f'Saved images for epoch {epoch} in section {section} at {save_path}')



if __name__ == "__main__":
    # Hyperparameters
    input_dim = 1
    out_dim = 128
    n_res_block = 2
    n_res_channel = 64
    stride = 2
    n_embed = 256
    embedding_dims = 128
    commitment_cost = 0.25
    num_epochs = 20
    learning_rate = 0.0003
    batch_size = 32

    train_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_train'
    test_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_test'
    val_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_validate'

    train_names =[os.path.join(train_image_directory, img) for img in os.listdir(train_image_directory) if img.endswith(('.nii', '.nii.gz'))]
    val_names= [os.path.join(val_image_directory, img) for img in os.listdir(val_image_directory) if img.endswith(('.nii', '.nii.gz'))]
    test_names= [os.path.join(test_image_directory, img) for img in os.listdir(test_image_directory) if img.endswith(('.nii', '.nii.gz'))]

    train_images = load_data_2D(train_names, normImage=True)
    val_images = load_data_2D(val_names, normImage=True)
    test_images = load_data_2D(test_names, normImage=True)


    # Create the VQVAE model
    model = VQVAE(input_dim, out_dim, n_res_block, n_res_channel, stride, n_embed, commitment_cost, embedding_dims)

    # Specify the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ssim_vals, ssim_vals_valid = train_vqvae(model, train_images, val_images, test_images, num_epochs, learning_rate, device, batch_size)


    # Save the model parameters after training
    torch.save(model.state_dict(), 'vqvae_model.pth')
    print("Model parameters saved as 'vqvae_model.pth'.")

    # Create a new VQVAE model instance for testing
    test_model_instance = VQVAE(input_dim, out_dim, n_res_block, n_res_channel, stride, n_embed, commitment_cost,
                                embedding_dims)
    test_model_instance.to(device)  # Move to the appropriate device

    # Load the saved model parameters
    test_model_instance.load_state_dict(torch.load('vqvae_model.pth'))
    print("Model parameters loaded.")

    # Create test DataLoader
    test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

    # Test the model on the test set
    test_model(test_model_instance, test_loader, device, ssim_metric)

    plot_ssims(ssim_vals, ssim_vals_valid, num_epochs, save_path='average_ssim_over_epochs.png')
