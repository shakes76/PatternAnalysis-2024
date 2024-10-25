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
from IPython.display import HTML
from tqdm import tqdm

import dataset
import modules
import utils
import predict
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Here")

#####################################################
# Training


# def train_stylegan2(discrim, gen, path_length_penalty, loader, opt_discrim, opt_gen, opt_mapping_network,):
#     """The training cycle of the StyleGAN2"""
#     loop = tqdm(loader, leave=True) # Visualises the training progress

#     curr_Gloss = []
#     curr_Dloss = []

#     for batch_idx, (real, _) in enumerate(loop):
#         real = real.to(device)
#         cur_batch_size = real.shape[0]

#         w = utils.get_w(cur_batch_size, modules.mapping_network, device)
#         noise = utils.get_noise(cur_batch_size, device)

#         # Use cuda AMP for accelerated training
#         with torch.amp.autocast('cuda'):
#             # Generate fake image using the Generator
#             fake = gen(w, noise)

#             # Get a critic score for the fake and real image
#             discrim_fake = discrim(fake.detach())
#             discrim_real = discrim(real)

#             # Calculate and log gradient penalty
#             gp = predict.gradient_penalty(discrim, real, fake, device=device)
#             loss_discrim = (
#                 -(torch.mean(discrim_real) - torch.mean(discrim_fake))
#                 + utils.lambda_gp * gp
#                 + (0.001 * torch.mean(discrim_real ** 2))
#             )
        
#         # Append the observed Discriminator loss to the list
#         curr_Dloss.append(loss_discrim.item())

#         # Reset gradients for the Discriminator
#         # Backpropagate the loss and update the discriminator's weights
#         discrim.zero_grad()
#         loss_discrim.backward()
#         opt_discrim.step()

#         # Get score for the Discriminator and the Generator
#         gen_fake = discrim(fake)
#         loss_gen = -torch.mean(gen_fake)

#         # Apply path length penalty on every 16 Batches
#         if batch_idx % 16 == 0:
#             plp = path_length_penalty(w, fake)
#             if not torch.isnan(plp):
#                 loss_gen = loss_gen + plp

#         # Append the observed Generator loss to the list
#         curr_Gloss.append(loss_gen.item())

#         modules.mapping_network.zero_grad() # Reset gradients for the mapping network and  generator
#         gen.zero_grad()
#         loss_gen.backward() # Backpropagate the generator loss
#         opt_gen.step()
#         opt_mapping_network.step()
#         # Update generator's weights and Update mapping network's weights
#         loop.set_postfix(
#             gp=gp.item(),
#             loss_critic=loss_discrim.item(),
#         )

#     return (curr_Dloss, curr_Gloss)


####################################################
# Main function (and training)

def gpu_warmup(device):
    if torch.cuda.is_available():
        # Create a small dummy tensor on the GPU
        dummy = torch.tensor([1.0], device=device)
        # Perform a simple operation to force CUDA context initialization
        dummy = dummy * 2
        print("GPU warmup complete")

#if __name__ == "main":
print("Here2")
# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

gpu_warmup(device) # Warm up the GPU to avoid errors

# Create batch of latent vectors used to visualise the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Dataloader
loader = dataset.get_data(data_root, log_resolution, batch_size)

gen = modules.Generator(log_resolution, w_dim).to(device)
discrim = modules.Discriminator(log_resolution).to(device)
mapping_network = modules.MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = modules.PathLengthPenalty(0.99).to(device)

# Initialise Adam optimiser
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_discrim = optim.Adam(discrim.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# Check if there are pre-trained models to load
load_model = False  # Set to True if you want to load a pre-trained model
if load_model:
    gen.load_state_dict(torch.load('/assets/netG.pth'))
    discrim.load_state_dict(torch.load('/assets/netD.pth'))
    mapping_network.load_state_dict(torch.load('/assets/netM.pth'))
    print("Loaded pre-trained models.")

if not load_model:
    # Train the following modules
    gen.train()
    discrim.train()
    mapping_network.train()

    # Keeps a Log of total loss over the training
    G_Loss = []
    D_Loss = []
    img_list = []

    # loop over total epcoh.
    for epoch in range(epochs):
        # curr_Gloss, curr_Dloss = train_stylegan2(
        #     discrim,
        #     gen,
        #     path_length_penalty,
        #     loader,
        #     opt_discrim,
        #     opt_gen,
        #     opt_mapping_network,
        # )

        #######################################################
        
        """The training cycle of the StyleGAN2"""
        loop = tqdm(loader, leave=True) # Visualises the training progress

        curr_Gloss = []
        curr_Dloss = []

        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)
            cur_batch_size = real.shape[0]

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

        ###########################################################

        # Append the current loss to the main list
        G_Loss.extend(curr_Gloss)
        D_Loss.extend(curr_Dloss)

        # Save generator's fake image on every 50th epoch
        if epoch % 10 == 0:
            predict.generate_examples(gen, mapping_network, epoch, device)

        if (epoch % 10 == 0) or ((epoch == epochs-1) and (epoch == len(loader)-1)):
            fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


    # Save the models after training
    torch.save(gen.state_dict(), '/assets/netG.pth')
    torch.save(discrim.state_dict(), '/assets/netD.pth')
    torch.save(mapping_network.state_dict(), '/assets/netM.pth')
    print("Saved models.")

    predict.plot_loss(G_Loss, D_Loss)

    # Create the animation
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


# import sklearn.datasets
# import pandas as pd
# import numpy as np
# import umap
# import umap.plot

# mapper = umap.UMAP().fit()
# umap.plot.points(mapper)


