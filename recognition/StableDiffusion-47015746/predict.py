"""
This file contains the image generation method given a trained Stable Diffusion mode.
"""



#s4701574

#CHATGPT was used to assist with writing some of this code


from modules import *
from dataset import *
import numpy as np
import PIL.Image as Image

def visualize_images(original_images,  step):
    """Takes in a batch of images, detaches them from cpu, unnormalizes, and then transposes them to prepare them for visualization"""

    # Move images back to CPU and detach
    original_images = original_images.cpu().detach()
    #reconstructed_from_latent = reconstructed_from_latent.cpu().detach()

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5
    
    original_images = unnormalize(original_images)
    return np.transpose(original_images[0].numpy(), (1, 2, 0))



#Source for "extract" and part of "reverse_diffusion": https://huggingface.co/blog/annotated-diffusion
timesteps = 1000
def extract(array, t, shape):
    """gets index "t" in array and reshapes it to shape"""

    batch_size = t.shape[0]  
    out = array.gather(0, t)  
    return out.view(batch_size, *((1,) * (len(shape) - 1)))  

@torch.no_grad()
def reverse_diffusion(model, x, t, t_index):
    """removes noise based on timestep and then adds a little back"""

    predicted_noise = model.unet(x, t)
    # Define beta schedule
    betas = model.noise_scheduler.betas

    # Define alphas 
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # Calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, predicted_noise.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, predicted_noise.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, predicted_noise.shape)

 
    # use our model to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)


    if t_index == 0:
        return model_mean
    else:
        posterior_var_t = extract(posterior_variance, t, predicted_noise.shape)
        noise1 = torch.randn_like(x)
        noise2 = torch.randn_like(x)
        return model_mean + 1.15*torch.sqrt(posterior_var_t) * (noise1 + noise2)



def create_gif(model, num_samples=1, gif_name="results.gif", shape=(1, 4, 32, 32), row = 3, column = 3):
    """Creates a gif of a grid of images being denoised (basically generates images)"""
   
    stepsize = int(10)
    frames = []
    
    with torch.no_grad():
        for i in range(row*column):
            x = torch.randn(shape, device=device)
            sub_frames = []
        
            for _, t in enumerate(reversed(list(range(1000)))):
                x = process_noise_with_model(x, model, t, num_samples, device)

                if t % stepsize == 0:
                    
                    
                    img_tensor = model.vqvae.decode(torch.clamp(x, -1, 1))
                    img = visualize_images(img_tensor, t)  # Get the image as a NumPy array
                    
                    # Convert the NumPy array to a PIL Image
                    img = Image.fromarray((img * 255).astype(np.uint8))  # Convert from [0, 1] to [0, 255] range
                    # Append the frame to the list
                    sub_frames.append(img)

            for i in range(5): #Extra iterations with extra layer of denoising
                for _, t in enumerate(reversed(list(range(30)))):
                    x = process_noise_with_model(x, model, t, num_samples, device)

                    if t % stepsize == 0:
                        time_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                        img_tensor = model.noise_scheduler.remove_noise(x, model.unet(x, time_tensor), time_tensor)
                        img_tensor = model.vqvae.decode(img_tensor)
                        # Visualize and process images using visualize_images
                        img = visualize_images(img_tensor, t)  # Get the image as a NumPy array
                        
                        # Convert the NumPy array to a PIL Image
                        img = Image.fromarray((img * 255).astype(np.uint8))  # Convert from [0, 1] to [0, 255] range

                        # Append the frame to the list
                        sub_frames.append(img)
            frames.append(sub_frames)
    create_gif_grid(frames, grid_size=(row,column), gif_path=gif_name)

def create_gif_grid(frame_lists, grid_size, gif_path, duration=100):
    """Does the plotting process for create_gif"""



    # Get frame size from the first frame in the first list
    frame_width, frame_height = frame_lists[0][0].size
    
    # Calculate the size of the grid image
    grid_width = frame_width * grid_size[1]
    grid_height = frame_height * grid_size[0]

    # Number of frames per section
    num_frames = len(frame_lists[0])

    # Prepare the list of frames for the final GIF
    gif_frames = []

    for i in range(num_frames):
        # Create a new blank image for the grid
        grid_image = Image.new("RGB", (grid_width, grid_height))

        # Loop through the sections in the grid
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                # Calculate the position for the current frame
                index = row * grid_size[1] + col
                if index < len(frame_lists):
                    # Get the frame for this section
                    current_frame = frame_lists[index][i]
                    # Paste the frame into the correct position in the grid
                    grid_image.paste(current_frame, (col * frame_width, row * frame_height))

        # Append the current grid image to the gif_frames list
        gif_frames.append(grid_image)

    # Save the frames as a GIF
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=duration, loop=1)

def process_noise_with_model(x, model, t, num_samples, device, shape=(1, 4, 32, 32)):
    """Process the noise tensor with the model for a single step."""

    time_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    x = reverse_diffusion(model, x, time_tensor, t)
    x = torch.clamp(x, -1, 1)

    return x
























