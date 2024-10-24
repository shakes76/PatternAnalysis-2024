'''
[desc]
contains some utility functions and class
for stable diffusion like noise schedular
and forward, backwards diffusion and sampling
for new image generation.

also contains image drawing function 

@author Jamie Westerhout
@project Stable Diffusion
@date 2024
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

class NoiseScheduler:
    '''
        class to determin the what level of noise
        to apply at each timestep
    '''
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.1):
        '''
            beta_start and beta_end: determin how much noise will be added
            nume_timestep: number of times steps to use
        '''
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

    def get_beta(self, t):
        '''
            returns bata at timestep t
        '''
        return self.betas[t]

    def get_alphas(self):
        '''
            alphas -> 1-betas
        '''
        return 1 - self.betas

    def get_alphas_cumprod(self):
        '''
            cumulaitve product of alphas
            used to calculate how much noise to add
        '''
        return torch.cumprod(self.get_alphas(), dim=0)
    
def add_noise(x, t, nosie_schedular):
    '''
        add noise to the images for timestep t
        where the noise is deterined by the nosie_schedular
    '''
    #create noise
    noise = torch.randn_like(x).to(device)

    #get data and calculate how much noise should be added to the images
    alpha_t = nosie_schedular.get_alphas_cumprod()[t].to(x.device)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_beta_t = torch.sqrt(1 - alpha_t)

    #add the noise to the image
    return sqrt_alpha_t * x + sqrt_beta_t * noise, noise

def reverse_diffusion(noisy_latent, predicted_noise, t, nosie_schedular):
    '''
        reversese the noise in noisy_letent at timestep t using the predicted noise
    '''

    # get the details about the noise at timestep t from the nosie_schedular
    alpha_t = nosie_schedular.get_alphas_cumprod()[t].to(noisy_latent.device)
    beta_t = nosie_schedular.get_beta(t).to(noisy_latent.device)
    alpha_prev = nosie_schedular.get_alphas_cumprod()[t - 1].to(noisy_latent.device) if t > 0 else torch.tensor(1.0).to(noisy_latent.device)

    # Remove the predicted noise from the noisy latent with respect to the given timestep
    denoised = (noisy_latent - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)

    if t > 0:
        # calculate variance so that it can be added back to the image
        # onyl when timestep is not zero since at 0 it needs to preocude a final
        # image
        variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
        z = torch.randn_like(noisy_latent)
        # return denoised image with noise scaled by varince added back
        return denoised + torch.sqrt(variance) * z
    else:
        #r eutrn denoised image
        return denoised
    
def display_images(images, num_images=5, title="Images"):
    '''
        simple function to plot num_images images from images
    '''
    images = images[:num_images].detach().cpu().numpy() 

    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = np.transpose(images[i], (1, 2, 0))
        axs[i].imshow(img, cmap = 'gray')
        axs[i].axis('off')
    fig.suptitle(title)
    plt.show()

def generate_sample(label, model, vae_decoder, vae_encoder, latent_dim, num_timesteps, nosie_schedular, num_samples=1):
    '''
        generats random noise in the size of the latent space
        then iterates throuhg all time steps denoising it at eachs step
        using the unet then decodes it using the unet
        this produces a completly new image conditioned
        by the unet 
    '''
    model.eval()
    vae_decoder.eval()
    output_images = torch.tensor(())

    #genete a tensor with the given label for the number fo samples
    label = torch.tensor([label] * num_samples).to(device)
    
    with torch.no_grad():
        #generate ranom latent
        x = torch.randn(num_samples, latent_dim).to(device)
        #turn latent form the 2 dim to 3 dim version
        x = vae_encoder.decode(x)
        for t in reversed(range(num_timesteps)):
            #predict noise for the timestep
            predicted_noise, _ = model(x, label, t)
            #remove noise for rhe timestep
            x = reverse_diffusion(x, predicted_noise, t, nosie_schedular)
            #decode into actual image
            output_image = vae_decoder(x)
            #time steps to give outputs for (all of them for 10 time steps)
            if t in [0,1,2,3,4,5,6,7,9,10]:
                output_images = torch.cat((output_images.to(device), output_image.to(device)), 0)

    return output_images

def generate_sample_latent(label, model, vae_decoder, vae_encoder, latent_dim, num_timesteps, num_samples=1):
    '''
        generats random noise in the size of the latent space
        then iterates throuhg all time steps getting the state
        of the data at the bottlenet after the cross attention step
    '''
    model.eval()
    vae_decoder.eval()
    output_images = torch.tensor(())

    #genete a tensor with the given label for the number fo samples
    label = torch.tensor([label] * num_samples).to(device)
    
    with torch.no_grad():
        #generate ranom latent
        x = torch.randn(num_samples, latent_dim).to(device)
        x = vae_encoder.decode(x)
        #turn increae form the 2 dim to 3 dim version
        for t in reversed(range(num_timesteps)):
            # get the latent tensor for the output of the bottleneck of the unet
            predicted_noise, y = model(x, label, t)
            output_image = y
            output_images = torch.cat((output_images.to(device), output_image.to(device)), 0)

    return output_images