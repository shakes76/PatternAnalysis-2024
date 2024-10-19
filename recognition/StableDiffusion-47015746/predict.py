from modules import *
from dataset import *
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import os
def visualize_images(original_images,  step):
    # Move images back to CPU and detach
    original_images = original_images.cpu().detach()
    #reconstructed_from_latent = reconstructed_from_latent.cpu().detach()

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5
    
    original_images = unnormalize(original_images)
    return np.transpose(original_images[0].numpy(), (1, 2, 0))
    
    #reconstructed_from_latent = unnormalize(reconstructed_from_latent)

    # Display first image (you can change num_images to display more)
    #num_images = 1
    #fig, axes = plt.subplots(1, num_images, figsize=(5, 5))

    # Transpose to change from (C, H, W) to (H, W, C)
    #axes.imshow(np.transpose(original_images[0].numpy(), (1, 2, 0)))
    #axes.axis('off')
    #axes.set_title("Original")

    # Reconstructed from Latent (Encoder → Decoder)
    #axes[1].imshow(np.transpose(reconstructed_from_latent[0].numpy(), (1, 2, 0)))
    #axes[1].axis('off')
    #axes[1].set_title("From Latent")

    #plt.suptitle(f"Step {step} - Model Outputs")
    #plt.show()








timesteps = 1000
def extract(array, t, shape):
    """
    Extracts the value at index `t` from `array` and reshapes it to match `shape`.
    
    Args:
        array (torch.Tensor): 1D tensor containing the values to extract from (e.g., betas, alphas).
        t (torch.Tensor): A tensor containing the current time step.
        shape (torch.Size): The shape to which the extracted values will be reshaped.
    
    Returns:
        torch.Tensor: The extracted values reshaped to the desired shape.
    """
    batch_size = t.shape[0]  # Get batch size from the timestep tensor
    out = array.gather(0, t)  # Gather values from the array at index `t`
    return out.view(batch_size, *((1,) * (len(shape) - 1)))  # Reshape to match `shape`

@torch.no_grad()
@torch.no_grad()
def reverse_diffusion_step(model, x, t, t_index):
    '''
    Sourced from: https://huggingface.co/blog/annotated-diffusion
    Author: Niels Rogge, Kashif Rasul
    '''
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
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

@torch.no_grad()
def image_generation(model, shape=(1, 128, 30, 32), save_path=None):
    device = next(model.parameters()).device

    # Setup figure for plotting
    fig = plt.figure(figsize=(15,15))
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")

    # Define reverse transformation for image normalization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5

    # Define grid dimensions for subplot (3x3 grid)
    rows = 3**2 
    cols = 3
    stepsize = int(timesteps/10)
    counter = 1

    for i in range(1, rows+1):
        # Initialize image with random noise
        img = torch.randn(shape, device=device)
        for j in reversed(range(0, timesteps)):
            t = torch.full((1,), j, device=device, dtype=torch.long)
            
            
            with torch.no_grad():
                # Perform a reverse diffusion step
                img= reverse_diffusion_step(model, img, t, j)
            # Plot the reconstructed image at specified timesteps
            if j % stepsize == 0:
                visualize_images(model.decoder(img), j)
                counter+=1



def create_gif(model, num_samples=1, frames_per_gif=100, gif_name="sampling.gif", shape=(1, 128, 30, 32)):
    stepsize = int(10)
    frames = []
    
    with torch.no_grad():
        x = torch.randn(shape, device=device)

        for _, t in enumerate(reversed(list(range(1000)))):
            x = process_noise_with_model(x, model, t, num_samples, device)

            if t % stepsize == 0:
                # Use model.decoder(x) to decode and get the image tensor
                img_tensor = model.decoder(x)
                
                # Visualize and process images using visualize_images
                img = visualize_images(img_tensor, t)  # Get the image as a NumPy array
                
                # Convert the NumPy array to a PIL Image
                img = Image.fromarray((img * 255).astype(np.uint8))  # Convert from [0, 1] to [0, 255] range

                # Append the frame to the list
                frames.append(img)

    # Save all frames as a GIF
    if frames:
        frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=100, loop=1)

    return x
k = 0
def process_noise_with_model(x, model, t, num_samples, device, shape=(1, 128, 30, 32)):
    """Process the noise tensor with the model for a single step."""
    time_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    eta_theta = model.unet(x, time_tensor)
    min = torch.min(eta_theta)
    max = torch.max(eta_theta)
    #visualize_images(model.decoder(eta_theta), 5)
    eta_theta = torch.clamp(eta_theta, -1, 1)

    betas = model.noise_scheduler.betas

    # Define alphas 
    alpha_t = extract(1 - betas, time_tensor, x.shape)  
    alpha_t_bar = torch.cumprod(alpha_t, axis=0)
    
    x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

    x = torch.clamp(x, -1, 1)

    # Debugging: Print intermediate values to track the growth of x
    #print(f"Step {t}: max(x)={x.max().item()}, min(x)={x.min().item()}")
    if t > 0:
        z = torch.randn(shape, device=device)
        beta_t = extract(betas, time_tensor, x.shape)
        sigma_t = beta_t.sqrt()
        #print(sigma_t)
        x = x + 0.5*sigma_t*z
        x = torch.clamp(x, -1, 1)

        #print(f"Step {t}: max(sigma_t * z)={torch.max(sigma_t * z).item()}, min(sigma_t * z)={torch.min(sigma_t * z).item()}")
    return x


if __name__ == '__main__':
    # Load the pre-trained VAE (encoder and decoder) and instantiate a new UNet and noise scheduler
    pre_trained_vae = VAE(in_channels=3, latent_dim=128, out_channels=3).to(device)
    pre_trained_vae.load_state_dict(torch.load("vae_state_dict.pth"))
    pre_trained_vae.eval()
    encoder = pre_trained_vae.encoder  # Use pre-trained encoder
    decoder = pre_trained_vae.decoder  # Use pre-trained decoder

    # Define the UNet and noise scheduler for the diffusion model
    unetF = UNet().to(device)
    noise_scheduler = NoiseScheduler(timesteps=1000)
    # Instantiate diffusion model with frozen encoder and decoder
    diffusion_model = DiffusionModel(encoder, unetF, decoder, noise_scheduler).to(device)
    diffusion_model.load_state_dict(torch.load("diffusion_model_state_dict.pth"))

    print("worked")

    #image_generation(diffusion_model)

    create_gif(diffusion_model)




















    # Load data
    data_train = "C:/Users/msi/Desktop/AD_NC/train" 
    data_test = "C:/Users/msi/Desktop/AD_NC/test" 
    #data_train = "/home/groups/comp3710/ADNI/AD_NC/train"
    #data_test = "/home/groups/comp3710/ADNI/AD_NC/test"
    #dataloader = load_data(data_train, data_test)
    

    #for images, labels in dataloader:
        #images = images.to(device)
        #print(images.shape)
        #t = torch.randint(0, diffusion_model.noise_scheduler.timesteps, (images.size(0),)).long().to(device)
        #latent = diffusion_model.encoder(images)
        #noisy_latent = diffusion_model.noise_scheduler.add_noise(latent, t)[0]
        #denoised_lated = diffusion_model.noise_scheduler.remove_noise(noisy_latent, diffusion_model.unet(noisy_latent,t), t)
        #visualize_images(diffusion_model.decoder(noisy_latent), 1)
        #visualize_images(diffusion_model.decoder(denoised_lated), 2)















    stepsize = int(50)
    img = torch.randn((1, 128, 30, 32), device=device)

    for j in reversed(range(0, 1000, 1)):

        print(j)
        # Set t to the current timestep
        t = torch.full((1,), j, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = diffusion_model.unet(img, t)
        
        # Remove and then add noise for reverse diffusion
        img = diffusion_model.noise_scheduler.remove_noise(img, predicted_noise, t)
        img1 = img
        img = diffusion_model.noise_scheduler.add_noise_partial(img, t, t)[0]


        # Visualize at certain steps
        if j % stepsize == 0:

            visualize_images(diffusion_model.decoder(img1), j)
         





