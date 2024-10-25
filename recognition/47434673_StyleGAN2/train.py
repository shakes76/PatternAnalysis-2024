from __future__ import print_function
#%matplotlib inline
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
from tqdm import tqdm

import dataset
import modules
import utils
import predict
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################################
# Main function (and training)

def gpu_warmup(device):
    """Warm up the GPU so no errors occur"""
    if torch.cuda.is_available():
        # Create a small dummy tensor on the GPU
        dummy = torch.tensor([1.0], device=device)
        # Perform a simple operation to force CUDA context initialization
        dummy = dummy * 2
        print("GPU warmup complete")

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

gpu_warmup(device) # Warm up the GPU to avoid errors

# Dataloader
loader = dataset.get_data(data_root, log_resolution, batch_size)

# Instantiate generator, discriminator and mapping network
gen = modules.Generator(log_resolution, w_dim).to(device)
discrim = modules.Discriminator(log_resolution).to(device)
mapping_network = modules.MappingNetwork(z_dim, w_dim).to(device)

# Intstantiate path length penalty for optimisation
path_length_penalty = modules.PathLengthPenalty(0.99).to(device)

# Initialise Adam optimisers
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_discrim = optim.Adam(discrim.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# Check if there are pre-trained models to load
if load_model:
    gen.load_state_dict(torch.load('/assets/netG.pth'))
    discrim.load_state_dict(torch.load('/assets/netD.pth'))
    mapping_network.load_state_dict(torch.load('/assets/netM.pth'))

    # Google colab residue
    # gen.load_state_dict(torch.load('/content/drive/My Drive/COMP3710/project/assets/netG.pth'))
    # discrim.load_state_dict(torch.load('/content/drive/My Drive/COMP3710/project/assets/netD.pth'))
    # mapping_network.load_state_dict(torch.load('/content/drive/My Drive/COMP3710/project/assets/netM.pth'))


    print("Loaded pre-trained models.")
    predict.generate_examples(gen, mapping_network, 14, device) # Change epoch from 14 to whatever epoch you want

if not load_model:
    # Train the following modules
    gen.train()
    discrim.train()
    mapping_network.train()

    # Keeps a Log of total loss over the training
    G_Loss = []
    D_Loss = []
    img_list = []

    # Training StyleGAN2
    for epoch in range(epochs):

        loop = tqdm(loader, leave=True) # Visualises the training progress

        curr_Gloss = []
        curr_Dloss = []
        
        # Training for each epoch (training for one iteration)
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Random noise
            w = utils.get_w(cur_batch_size, mapping_network, device)
            noise = utils.get_noise(cur_batch_size, device)

            # Use cuda AMP for accelerated training
            with torch.amp.autocast('cuda'):
                # Generate fake image using the Generator
                fake = gen(w, noise)

                # Get a critic score for the fake and real image
                discrim_fake = discrim(fake.detach())
                discrim_real = discrim(real)

                # Calculate and log gradient penalty
                gp = utils.gradient_penalty(discrim, real, fake, device=device)
                loss_discrim = (
                    -(torch.mean(discrim_real) - torch.mean(discrim_fake))
                    + utils.lambda_gp * gp
                    + (0.001 * torch.mean(discrim_real ** 2))
                )
            
            # Append the observed Discriminator loss to the list
            curr_Dloss.append(loss_discrim.item())

            # Reset gradients for the Discriminator
            # Backpropagate the loss and update the discriminator's weights
            discrim.zero_grad()
            loss_discrim.backward()
            opt_discrim.step()

            # Get score for the Discriminator and the Generator
            gen_fake = discrim(fake)
            loss_gen = -torch.mean(gen_fake)

            # Apply path length penalty on every 16 Batches
            if batch_idx % 16 == 0:
                plp = path_length_penalty(w, fake)
                if not torch.isnan(plp):
                    loss_gen = loss_gen + plp

            # Append the observed Generator loss to the list
            curr_Gloss.append(loss_gen.item())

            mapping_network.zero_grad() # Reset gradients for the mapping network and  generator
            gen.zero_grad()
            loss_gen.backward() # Backpropagate the generator loss
            opt_gen.step()
            opt_mapping_network.step()
            # Update generator's weights and Update mapping network's weights
            loop.set_postfix(
                gp=gp.item(),
                loss_critic=loss_discrim.item(),
            )

        # Append the current loss to the main list
        G_Loss.extend(curr_Gloss)
        D_Loss.extend(curr_Dloss)

        # Save generator's fake image on every 10th epoch
        if epoch % 10 == 0:
            predict.generate_examples(gen, mapping_network, epoch, device)


    # Save the models after training
    torch.save(gen.state_dict(), '/assets/netG.pth')
    torch.save(discrim.state_dict(), '/assets/netD.pth')
    torch.save(mapping_network.state_dict(), '/assets/netM.pth')
    print("Saved models.")

    # Plot the loss graphs
    predict.plot_loss(G_Loss, D_Loss)






