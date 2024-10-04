# In your train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchmetrics.functional.image \
    import structural_similarity_index_measure as ssim
import numpy as np
import matplotlib.pyplot as plt
import os



from dataset import HipMRILoader
import modules


# Hyperparameters
num_epochs = 3
batch_size = 32
lr = 0.0002
num_hiddens = 128
num_residual_hiddens = 32
num_channels = 1
num_embeddings = 512
dim_embedding = 64
beta = 0.25

# Configure Pytorch
seed = 42
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)), 
])

# Get loaders (variance == 5)
train_loader, validate_loader, data_variance = HipMRILoader(
    train_dir, validate_dir, test_dir,
    batch_size=batch_size, transform=transform
    ).get_loaders()



# Create model
model = modules.VQVAE(
    num_channels=num_channels,
    num_hiddens=num_hiddens,
    num_residual_hiddens=num_hiddens,
    num_embeddings=num_embeddings,
    dim_embedding=dim_embedding,
    beta=beta)

model = model.to(device)

# Set optimiser
optimizer = optim.Adam(
    model.parameters(), 
    lr=lr,
    amsgrad=False)

# Training mectrics
epoch_training_output_loss = []
epoch_training_vq_loss = []

epoch_validation_output_loss = []
epoch_validation_vq_loss = []
epoch_ssim = []

# Directory for saving images
save_dir = 'saved_images'

# Training loop
for epoch in range(num_epochs):
    model.train()
    training_output_error = []
    training_vq_error = []

    for i, training_images in enumerate(train_loader):
        training_input_images = training_images.to(device)

        optimizer.zero_grad()
        vq_loss, training_output_images = model(training_input_images)

        # Calculate reconstruction loss
        output_loss = F.mse_loss(training_output_images, training_input_images) / data_variance
        loss = output_loss + vq_loss
        loss.backward()

        optimizer.step()

        training_output_error.append(output_loss.item())
        training_vq_error.append(vq_loss.item())
    
    # Calculate and store average training losses
    training_output_loss = np.mean(training_output_error)
    epoch_training_output_loss.append(training_output_loss)
    training_vq_loss = np.mean(training_vq_error)
    epoch_training_vq_loss.append(training_vq_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training output loss: {training_output_loss:.5f}, Training VQ loss: {training_vq_loss:.5f}')

    # Evaluate on the validation dataset
    model.eval()
    validation_output_error = []
    validation_vq_error = []
    validation_ssim = []

    with torch.no_grad():
        for j, validation_images in enumerate(validate_loader):
            validation_input_images = validation_images.to(device)
            validation_vq_loss, validation_output_images = model(validation_input_images)

            # Reshape images for SSIM calculation
            real_image = validation_input_images.view(-1, 1, 256, 128).detach().to(device)
            decoded_image = validation_output_images.view(-1, 1, 256, 128).detach().to(device)
            
            # Calculate SSIM and store it
            similarity = ssim(decoded_image, real_image, data_range=1.0).item()
            validation_ssim.append(similarity)

             # Calculate output loss for validation
            validation_output_loss = F.mse_loss(validation_output_images, validation_input_images) / data_variance
            validation_output_error.append(validation_output_loss.item())
            validation_vq_error.append(validation_vq_loss.item())

    # Average validation loss and SSIM
    average_validation_loss = np.mean(validation_output_error)
    epoch_validation_output_loss.append(average_validation_loss)
    average_validation_vq_loss = np.mean(validation_vq_error)
    epoch_validation_vq_loss.append(average_validation_vq_loss)
    average_ssim = np.mean(validation_ssim)
    epoch_ssim.append(average_ssim)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation output loss: {average_validation_loss:.5f}, Validation VQ loss: {average_validation_vq_loss}, Average SSIM: {average_ssim:.5f}')
    print()

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


epochs = range(1, num_epochs + 1)

# 1. Plot Training Output Loss vs. Validation Output Loss
plt.figure(figsize=(12, 6))  # Width of normal page
plt.plot(epochs, epoch_training_output_loss, label='Training Output Loss')
plt.plot(epochs, epoch_validation_output_loss, label='Validation Output Loss')
plt.xlabel('Epochs')
plt.ylabel('Output Loss')
plt.title('Training and Validation Output Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'output_loss_per_epoch.png'))
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
combined_training_loss = [x + y for x, y in zip(epoch_training_output_loss, epoch_training_vq_loss)]
combined_validation_loss = [x + y for x, y in zip(epoch_validation_output_loss, epoch_validation_vq_loss)]

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

# Define the save directory and ensure it exists
model_dir = 'saved_model/model.pth'
os.makedirs(os.path.dirname(model_dir), exist_ok=True)

# Save the model state_dict
torch.save(model.state_dict(), model_dir)