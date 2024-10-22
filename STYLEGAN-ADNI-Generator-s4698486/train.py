# train.py
import torch.amp
import torch
from torch import optim
from tqdm import tqdm

import generate_images
from constants import *
from dataset import get_data
from modules import *
import utils
from torch.optim.lr_scheduler import CyclicLR

# SETTING UP

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# Module initilization
loader = get_data(DATA, batch_size)

# Obtain models from modules.py.
generator = Generator(log_resolution, w_dim).to(device)
discriminator = Discriminator(log_resolution).to(device)
mapping_network = MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = PathLengthPenalty(0.99).to(device)

# Initilise Adam optimisation for all modules.
generator_optimiser = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.99)) # NOTE: Parameters according to StyleGAN2 paper.
discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
mapping_network_optimiser = optim.Adam(mapping_network.parameters(), lr=base_lr, betas=(0.0, 0.99))

# Use a learning rate scheduler to increase learning rate of generator and discriminator far beyond that of the mapping network
# which is suggested by the StyleGAN2 paper.
generator_scheduler = CyclicLR(generator_optimiser, base_lr=base_lr, max_lr=learning_rate, step_size_up=2000, mode='triangular')
discriminator_scheduler = CyclicLR(discriminator_optimiser, base_lr=base_lr, max_lr=learning_rate, step_size_up=2000, mode='triangular')


'''
Main training loop
'''
def train_fn():
    loop = tqdm(loader, leave=True) # Adds a progress bar for training.

    current_gen_loss = []
    current_discriminator_loss = []

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        style_vector = utils.get_style_vector(cur_batch_size, mapping_network, device)
        noise = utils.get_noise(cur_batch_size, device)

        # Use cuda AMP for accelerated training
        with torch.amp.autocast(device_type=device):
            generated_images = generator(style_vector, noise)
            discriminator_fake_output = discriminator(generated_images.detach()) # Will be our predicted labels for each of the fake images.
            
            # So, real is [32, 1, 256, 256]
            discriminator_real_output = discriminator(real) # Will be our predicted labels for each of the real images.

            # Combining real and fake leads to [32, 32, 256, 256] sized interpolated images.
            gp = utils.gradient_penalty(discriminator, real, generated_images, device=device)

            discriminator_loss = utils.discriminator_loss(real_output=discriminator_real_output, fake_output=discriminator_fake_output)
            generator_loss = utils.generator_loss(fake_output=discriminator_fake_output)

            # Update discriminator loss with gradient penalty and regularisation.
            discriminator_loss += lambda_gp * gp
            discriminator_loss += 0.001 * torch.mean(discriminator_real_output ** 2)

            regularization = 0.001 * torch.mean(torch.square(list(generator.parameters())[0])) # Adjust the regularization strength
            generator_loss += regularization
        
        # Append the observed Discriminator loss to the list
        current_discriminator_loss.append(discriminator_loss.item())

        # Backpropagate discriminator - letting it learn.
        discriminator.zero_grad()
        discriminator_loss.backward()
        discriminator_optimiser.step()
        discriminator_scheduler.step()

        # Apply path length penalty on every 16 Batches
        if batch_idx % 16 == 0:
            plp = path_length_penalty(style_vector, generated_images)
            if not torch.isnan(plp):
                # Trying to minimise plp's potential to cause nans.
                plp = torch.clamp(plp, min=0.1, max=10.0) # plp is being clipped basically every time...
                generator_loss = generator_loss + plp

        # Append the observed Generator loss to the list
        current_gen_loss.append(generator_loss.item())
       
        # Learning for generator and mapping network.
        mapping_network.zero_grad()
        generator.zero_grad()
        generator_loss.backward()
        generator_optimiser.step()
        generator_scheduler.step()
        mapping_network_optimiser.step()

        # Updates progress whilst training.
        loop.set_postfix(
            loss_critic=discriminator_loss.item(),
            loss_gen=generator_loss.item(),
        )

    return (current_discriminator_loss, current_gen_loss)
 
if __name__ == "__main__":

    # Train the following modules
    generator.train()
    discriminator.train()
    mapping_network.train()

    # Keeps a Log of total loss over the training
    lifetime_generator_losses = []
    lifetime_discriminator_losses = []

    # loop over total epcoh.
    for epoch in range(epochs):
        current_discriminator_loss, current_gen_loss = train_fn()

        # Append the current loss to the main list
        lifetime_generator_losses.extend(current_gen_loss)
        lifetime_discriminator_losses.extend(current_discriminator_loss)

        # Save generator's fake image every 2nd epoch
        if epoch % 2 == 0:
            generate_images.generate_examples(generator, mapping_network, epoch, device)

	    # Save model every 4 epochs
        if epoch % 4 == 0:
            torch.save(generator.state_dict(), f'Models/Gen/{image_height}x{image_width}/{epoch}')
            torch.save(discriminator.state_dict(), f'Models/Discriminator/{image_height}x{image_width}/{epoch}')
            torch.save(mapping_network.state_dict(), f'Models/MappingNetwork/{image_height}x{image_width}/{epoch}')


    # Once training complete, plot lifetime generator and discriminator losses.
    generate_images.plot_loss(lifetime_generator_losses, lifetime_discriminator_losses)
