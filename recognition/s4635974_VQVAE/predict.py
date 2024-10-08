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
    
    model = torch.load(model_path)
    print('Using saved model\n')

    model.to(device)
    model.eval()
    
    # Directories for datasets
    train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
    test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
    validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

    # Define your transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  
        # transforms.Normalize((0.5,), (0.5,)),
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
            # Run batch through model
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

    # Visulise 4 random images through the model and save in test_save_dir
    # Select 4 random indices from the test set
    random_indices = random.sample(range(len(test_loader.dataset)), 4)

    # Create a figure to plot the images
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))

    for idx, rand_idx in enumerate(random_indices):
        # Get the real image and pass it through the model to get the decoded image
        real_image = test_loader.dataset[rand_idx]
        real_image = real_image.unsqueeze(0).to(device)  # Add batch dimension and send to device
        _, decoded_image = model(real_image)

        # Detach and move to CPU for visualization
        real_image = real_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Squeeze and rearrange for plotting
        decoded_image = decoded_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

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

if (__name__ == "__main__"):
    predict(model_path=model_path)




    



            


