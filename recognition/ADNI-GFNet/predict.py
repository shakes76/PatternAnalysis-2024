# Import necessary modules and functions from the 'train' module 
from train import * 
# Import other necessary libraries
from torchvision.utils import save_image
import os

# Function to generate example images using a generator model
def generate_examples(gen, epoch, n=20):
    for epoch in range(EPOCHS):
        if epoch % 20 == 0:
            # Set the generator model in evaluation mode
            gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)  # Initialize generator  # Create an instance of your model
            gen.load_state_dict(torch.load(f'generator_epoch{epoch}.pt'))  # Load the saved state dictionary
            gen.eval()
            
            # Generate 'n' example images
            for i in range(n):
                with torch.no_grad():
                    w_values = [1, 2, 3, 4, 5]
                    for value in w_values:
                        # Generate random latent vector 'w'
                        w = get_w(value)
                        
                        # Generate random noise
                        noise = get_noise(1)
                        
                        # Generate an image using the generator model
                        img = gen(w, noise)
                        
                        # Create a directory to save the images for the current epoch if it doesn't exist
                        if not os.path.exists(f'generated_images/epoch{epoch}/w{value}'):
                            os.makedirs(f'generated_images/epoch{epoch}/w{value}')
                        
                        # Save the generated image with appropriate scaling
                        save_image(img*0.5+0.5, f"generated_images/epoch{epoch}/w{value}/img_{i}.png")
