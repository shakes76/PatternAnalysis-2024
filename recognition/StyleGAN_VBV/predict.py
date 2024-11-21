import torch
from torchvision.utils import save_image, make_grid
import os 

# Constants for model dimensions and device configuration
Z_DIM = 512  # Dimensionality of the latent space
W_DIM = 512  # Dimensionality for the style space
IN_CHANNELS = 512  # Number of input channels for the generator
CHANNELS_IMG = 3  # Number of output channels for images (RGB)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU
AD_IMAGES = "AD_"
NC_IMAGES = "NC_"
CURRENT_DATASET = NC_IMAGES


def generate_examples(gen, steps, n=9):
    """
    Generate and save n example images in a grid of size 3x3 using the trained generator model.
    """
    gen.eval()  # Set the generator to evaluation mode
    alpha = 1.0  # Set alpha to 1 for full resolution generation

    # Create a directory for saving images
    save_dir = f'saved_examples/step{steps}'
    os.makedirs(save_dir, exist_ok=True)

    images = []  # List to hold generated images

    for _ in range(n):  # Generate 'n' images
        with torch.no_grad():  # Disable gradient calculation for inference
            # Generate random noise input for the generator
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            # Generate an image from the noise input using the generator
            img = gen(noise, alpha, steps)
            images.append(img)   

    # Create a grid of images (3x3 for 9 images)
    grid = make_grid(torch.cat(images, dim=0), nrow=3)

    # Save the generated grid image; rescale to [0, 1] for saving
    save_image(grid * 0.5 + 0.5, os.path.join(save_dir, CURRENT_DATASET + f"grid_step{steps}.png"))
    
    gen.train()  # Set the generator back to training mode


def generate_examples_seperate(gen, steps, n=100):
    """
    Generate and save example images using the trained generator model.
    """
    gen.eval()  # Set the generator to evaluation mode
    alpha = 1.0  # Set alpha to 1 for full resolution generation

    # Create a directory for saving images
    save_dir = f'saved_examples_seperate/{CURRENT_DATASET}'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n):  # Generate 'n' images
        with torch.no_grad():  # Disable gradient calculation for inference
            # Generate random noise input for the generator
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            # Generate an image from the noise input using the generator
            img = gen(noise, alpha, steps)

            # Save each image separately
            img_path = os.path.join(save_dir, CURRENT_DATASET+f"image_{i+1}_step{steps}.png")
            save_image(img * 0.5 + 0.5, img_path)  # Rescale to [0, 1] for saving

    gen.train()  # Set the generator back to training mode

