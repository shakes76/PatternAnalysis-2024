"""
=======================================================================
File Name: predict.py
Author: Baibhav Mund
Student ID: 48548700
Description:
    This script is used to generate example images using a pre-trained StyleGAN2 generator 
    model. It loads a saved generator checkpoint (based on the specified epoch), 
    generates images based on randomly sampled latent vectors and noise, and saves 
    the generated images to a directory. The images are saved for different latent 
    vector (w-values) and epochs, allowing for a visual inspection of the model's 
    output.

    Key Features:
    - Command-line interface for specifying the epoch of the generator checkpoint to load.
    - Generates a specified number of images (`n` images) using latent vectors (`w-values`)
      and random noise inputs.
    - Saves the generated images to a directory structure organized by epoch and w-values.
    - The images are scaled and saved using `torchvision` utilities for easy visualization.

Usage:
    1. Set the desired epoch to load the generator checkpoint by passing the `--epoch` argument.
    2. Optionally, specify the number of images to generate using the `--n` argument (default: 10).
    3. Run the script using the command:
       `python predict.py --epoch 45 --n 5`
    4. The generated images will be saved in the `generated_images/epochX/wY/` folder 
       (where X is the epoch and Y is the w-value used for generation).

Dependencies:
    - PyTorch: Install via `pip install torch torchvision`
    - argparse: For command-line argument parsing (included in Python's standard library).
    - os: For directory creation and file management.
    - torchvision: For image saving utilities.

Functions:
    - generate_examples(): Generates and saves example images using the generator model.
    - Main block: Parses command-line arguments and calls `generate_examples()`.

Parameters:
    --epoch: Specifies the epoch number of the generator model checkpoint to load.
    --n: Specifies the number of images to generate for each w-value (default: 10).

Output:
    - The generated images are saved in a directory structure organized by epoch and w-values:
      `generated_images/epochX/wY/img_Z.png`
      - X is the epoch, Y is the w-value used for generation, Z is the image index.

=======================================================================
"""

from train import Generator, get_w, get_noise, LOG_RESOLUTION,W_DIM,DEVICE 
import torch
from torchvision.utils import save_image
import os
import argparse

# Function to generate example images using a generator model
def generate_examples(gen, epoch, n=10):
    # Load the generator's saved state for the specified epoch
    checkpoint_path = f'generator_epoch{epoch}.pt'
    
    if os.path.exists(checkpoint_path):
        # Set the generator model in evaluation mode
        gen.load_state_dict(torch.load(checkpoint_path))  # Load the saved state dictionary
        gen.eval()
        
        # Generate 'n' example images
        with torch.no_grad():
            w_values = [1, 2, 3, 4, 5]  # Example latent w-values to use
            for i in range(n):
                for value in w_values:
                    # Generate random latent vector 'w'
                    w = get_w(value)  # Assuming this function generates the latent vector
                    
                    # Generate random noise
                    noise = get_noise(1)  # Assuming this function generates the noise
                    
                    # Generate an image using the generator model
                    img = gen(w, noise)
                    
                    # Create a directory to save the images for the current epoch and w-value
                    img_dir = f'generated_images/epoch{epoch}/w{value}'
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    
                    # Save the generated image with appropriate scaling
                    save_image(img * 0.5 + 0.5, f"{img_dir}/img_{i}.png")
    else:
        print(f"Checkpoint for epoch {epoch} not found.")

# Main block to handle command-line arguments
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate images from a saved generator model checkpoint")
    parser.add_argument('--epoch', type=int, required=True, help='The epoch of the generator to load')
    parser.add_argument('--n', type=int, default=10, help='Number of images to generate')
    
    args = parser.parse_args()
    
    # Initialize generator 
    gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)

    # Call the generate_examples function
    generate_examples(gen, epoch=args.epoch, n=args.n)