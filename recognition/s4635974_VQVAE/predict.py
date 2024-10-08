import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchmetrics.functional.image \
    import structural_similarity_index_measure as ssim
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import train
from dataset import HipMRILoader
import modules


# Model path if using existing model
model_path = 'saved_model/lr=0.003.pth'

# Hyperparmeters if training again
num_epochs = 3
batch_size = 16
lr = 0.003
num_hiddens = 128
num_residual_hiddens = 32
num_channels = 1
num_embeddings = 512
dim_embedding = 64
beta = 0.25

# Save directory if training again
train_save_image_dir = f'train_images'

# Model save diretory if training again
train_save_model_dir = f'saved_model/train.pth'

# Retrain bool. Set to true if retaining
retrain_model = False

# Directory for saving test images
test_save_dir = f'test_images'
os.makedirs(test_save_dir, exist_ok=True)

def predict(
        model_path=model_path, 
        retrain_model=retrain_model 
        ):
    
    # Configure Pytorch
    seed = 42
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    print()
    
    # Create a new instance of your model
    model = modules.VQVAE(
        num_channels=num_channels,
        num_hiddens=num_hiddens,
        num_residual_hiddens=num_hiddens,
        num_embeddings=num_embeddings,
        dim_embedding=dim_embedding,
        beta=beta).to(device)

    # Load the saved state_dict into the model
    model.load_state_dict(torch.load(model_path))
    print('Using saved model\n')

    # model.to(device)
    model.eval()
    
    # Directories for datasets
    train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
    test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
    validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

    # Define your transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0,), (1,)) 
    ])

    # Get test loader
    test_loader = HipMRILoader(
        train_dir, validate_dir, test_dir,
        batch_size=batch_size, transform=transform
        ).get_test_loader()
    
    # Save average SSIM per batch
    SSIM  = []

    # Test loop
    with torch.no_grad():

        batch_SSIM = []

        for i, training_images in enumerate(test_loader):

            training_images = training_images.to(device)
            _, test_output_images = model(training_images)

            # Reshape images for SSIM calculation
            real_image = training_images.view(-1, 1, 256, 128).detach()
            decoded_image = test_output_images.view(-1, 1, 256, 128).detach()
            
            # Calculate SSIM and store it
            similarity = ssim(decoded_image, real_image, data_range=1.0).item()
            batch_SSIM.append(similarity)
        
        # Calculate average SSIM per batch
        batch_ssim = np.mean(batch_SSIM)
        SSIM.append(batch_ssim)

    average_SSIM = np.mean(SSIM)
    print("Average SSIM on training set: ", average_SSIM)

    # Visualize 4 random images through the model and save in test_save_dir
    # Create a figure to plot the images
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))

    # Loop through random indices to visualize
    for idx, real_image in enumerate(test_loader):  # Get batch from DataLoader
        if idx >= 4:  # We only need 4 images
            break

        # Ensure the real_image is on the device
        real_image = real_image.to(device)

        # Pass it through the model to get the decoded image
        _, decoded_image = model(real_image)

        # Detach and move to CPU for visualization (select only the first in the batch)
        real_image = real_image[0].cpu().numpy().squeeze()  # Assuming it's a single-channel image
        decoded_image = decoded_image[0].cpu().detach().numpy().squeeze()

        # Plot real image on the left
        axes[idx, 0].imshow(real_image, cmap='gray')
        axes[idx, 0].set_title('Real Image')
        axes[idx, 0].axis('off')

        # Plot decoded image on the right
        axes[idx, 1].imshow(decoded_image, cmap='gray')
        axes[idx, 1].set_title('Decoded Image')
        axes[idx, 1].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(test_save_dir, 'real_vs_decoded.png'))
    plt.show()
    
    print("End")

if (__name__ == "__main__"):
    predict(model_path=model_path)




    



            


