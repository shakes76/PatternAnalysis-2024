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
    
    # Initialize your generator (adjust LOG_RESOLUTION and W_DIM accordingly)
    gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)

    # Call the generate_examples function
    generate_examples(gen, epoch=args.epoch, n=args.n)