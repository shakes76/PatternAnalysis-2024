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
num_epochs = 5
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
print(device)

# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)), 
    # transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.RandomRotation(15),  # Random rotation within 15 degrees
])

# Get loaders
train_loader, validate_loader, data_variance = HipMRILoader(
    train_dir, validate_dir, test_dir,
    batch_size=batch_size, transform=transform
    ).get_loaders()

# Create model
model = modules.VQVAE(
    num_channels=num_channels,
    num_hiddens=num_hiddens,
    num_embeddings=num_embeddings,
    dim_embedding=dim_embedding,
    beta=beta)

# Set optimiser
optimizer = optim.Adam(
    model.parameters(), 
    lr=lr,
    amsgrad=False)

# Training mectrics
epoch_training_loss = []
epoch_validation_loss = []
epoch_ssim = []

# Directory for saving images
save_dir = 'saved_images'

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    model.train()
    training_error = []

    for i, training_images in enumerate(train_loader):
        training_input_images = training_images.to(device)
        optimizer.zero_grad()
        vq_loss, training_output_images = model(training_input_images)

        # Calculate reconstruction loss
        output_loss = F.mse_loss(training_output_images, training_input_images) / data_variance
        loss = output_loss + vq_loss
        loss.backward()

        optimizer.step()

        training_error.append(output_loss.item())
    
    # Calculate and store average training loss
    training_loss = np.mean(training_error[-300:])
    epoch_training_loss.append(training_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {training_loss:.5f}')

    # Evaluate on the validation dataset
    model.eval()
    validation_loss = 0
    validation_ssim = []
    with torch.no_grad():
        for j, validation_images in enumerate(validate_loader):
            validation_input_images = validation_images.to(device)
            vq_loss, validation_output_images = model(validation_input_images)

            # Reshape images for SSIM calculation
            real_image = validation_input_images.view(-1, 1, 128, 128).detach().to(device)
            decoded_image = validation_output_images.view(-1, 1, 128, 128).detach()
            
            # Calculate SSIM and store it
            similarity = ssim(decoded_image, real_image, data_range=1.0).item()
            validation_ssim.append(similarity)

             # Calculate output loss for validation
            output_loss = F.mse_loss(validation_output_images, validation_input_images) / data_variance
            validation_loss += output_loss + vq_loss

    # Average validation loss and SSIM
    average_validation_loss = validation_loss / len(validate_loader)
    epoch_validation_loss.append(average_validation_loss.item())
    average_ssim = np.mean(validation_ssim)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation loss: {average_validation_loss:.5f}, Average SSIM: {average_ssim:.5f}')
    epoch_ssim.append(average_ssim)
    print()

    # Save a few real and decoded images for inspection
    num_images_to_save = 5  
    real_images = validation_input_images.cpu().numpy()
    decoded_images = validation_output_images.cpu().numpy()  

    # Save images
    for k in range(num_images_to_save):
        # Save real images
        plt.imshow(real_images[k, 0], cmap='gray')
        plt.title('Real Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'real_image_{epoch + 1}_{k + 1}.png'))
        plt.close()

        # Save decoded images
        plt.imshow(decoded_images[k, 0], cmap='gray')
        plt.title('Decoded Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'decoded_image_{epoch + 1}_{k + 1}.png'))
        plt.close()


# Plotting and saving the graphs
plt.figure(figsize=(12, 6))

# Plot Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epoch_training_loss, label='Training Loss', color='blue')
plt.plot(epoch_validation_loss, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the Loss plot
plt.savefig(os.path.join(save_dir, 'training_validation_loss.png'))

# Plot Average SSIM
plt.subplot(1, 2, 2)
plt.plot(epoch_ssim, label='Average SSIM', color='green')
plt.title('Average SSIM over Epochs')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()

# Save the SSIM plot
plt.savefig(os.path.join(save_dir, 'average_ssim.png'))

# Close the figure to free up memory
plt.close()

# Save model
torch.save(model, save_dir)