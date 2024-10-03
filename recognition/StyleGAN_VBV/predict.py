import torch
from torchvision.utils import save_image 
from modules import Generator
import os 

# Constants for model dimensions and device configuration
Z_DIM = 512  # Dimensionality of the latent space
W_DIM = 512  # Dimensionality for the style space
IN_CHANNELS = 512  # Number of input channels for the generator
CHANNELS_IMG = 3  # Number of output channels for images (RGB)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU

def generate_examples(gen, steps, n=100):
    """
    Generate and save example images using the trained generator model.

    Parameters:
        gen (Generator): The trained generator model.
        steps (int): The current training step (resolution) of the generator.
        n (int): Number of images to generate and save.
    """
    gen.eval()  # Set the generator to evaluation mode
    alpha = 1.0  # Set alpha to 1 for full resolution generation

    for i in range(n):  # Loop to generate 'n' images
        with torch.no_grad():  # Disable gradient calculation for inference
            # Generate random noise input for the generator
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            # Generate an image from the noise input using the generator
            img = gen(noise, alpha, steps)

            # Create a directory for saving images if it doesn't exist
            if not os.path.exists(f'saved_examples/step{steps}'):
                os.makedirs(f'saved_examples/step{steps}')

            # Save the generated image; rescale to [0, 1] for saving
            save_image(img * 0.5 + 0.5, f"saved_examples/step{steps}/img_{i}.png")
    
    gen.train()  # Set the generator back to training mode
