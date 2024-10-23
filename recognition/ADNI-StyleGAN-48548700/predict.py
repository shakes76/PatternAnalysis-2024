import torch
import os
from torchvision.utils import save_image
from train import * 

# Function to generate example images using the pre-trained generator model
def generate_examples(epoch, n=20):
    # Initialize and load the generator model for the specified epoch
    gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)  # Create an instance of your generator model
    gen.load_state_dict(torch.load(f'generator_epoch{epoch}.pt', map_location=DEVICE))  # Load the pre-trained state dictionary
    gen.eval()  # Set generator model to evaluation mode
    
    # Generate 'n' example images
    with torch.no_grad():  # Disable gradient calculation
        for i in range(n):
            w_values = [1, 2, 3, 4, 5]  # Example latent vector scale values
            for value in w_values:
                w = get_w(value)  # Generate latent vector 'w'
                noise = get_noise(1)  # Generate random noise
                img = gen(w, noise)  # Generate an image using the generator model

                # Create a directory to save the images for the current epoch
                save_dir = f'generated_images/epoch{epoch}/w{value}'
                os.makedirs(save_dir, exist_ok=True)

                # Save the generated image with appropriate scaling (from [-1, 1] to [0, 1])
                save_image(img * 0.5 + 0.5, f"{save_dir}/img_{i}.png")

# Example usage:
if __name__ == "__main__":
    # Generate images every 20 epochs (or modify according to your need)
    for epoch in range(EPOCHS):
        generate_examples(epoch, n=10)  # Generate 10 example images for each set of latent vectors

