"""
train.py created by Matthew Lockett 46988133
"""
import random
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import hyperparameters as hp
from dataset import load_ADNI_dataset
from modules import *

# Force the creation of a folder to save figures if not present 
os.makedirs(hp.SAVED_FIGURES_DIR, exist_ok=True)

# PyTorch Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU.")

# Set random seed for reproducibility
print("Random Seed Used: ", hp.RANDOM_SEED)
random.seed(hp.RANDOM_SEED)
torch.manual_seed(hp.RANDOM_SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Load the ADNI dataset images for training
train_loader = load_ADNI_dataset()

# Plot a sample of images from the ADNI dataset saved as 'adni_sample_images.png'
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Training Images for the ADNI Dataset")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(hp.SAVED_FIGURES_DIR, "adni_sample_images.png"), bbox_inches='tight', pad_inches=0)
plt.close()

# Setup both the generator and discriminator models
gen = Generator().to(device)
disc = Discriminator().to(device)

# Create the optimisers used by the genertator and discriminator during training 
gen_opt = optim.Adam(gen.parameters(), lr=hp.LEARNING_RATE, betas=(0.0, 0.99))
disc_opt = optim.Adam(disc.parameters(), lr=hp.LEARNING_RATE, betas=(0.0, 0.99))

# Use Binary Cross Entropy Loss (BCELoss)
criterion = nn.BCELoss()

# Initialise the real and fake labels for training
real_label = 1.0
fake_label = 0.0

# Create a collection of latent vectors used later to visualise the progression of the generator 
fixed_noise = torch.randn(64, hp.LATENT_SIZE, device=device)

# Lists to keep track of progress and statistics 
img_list = []
G_losses = []
D_losses = []
iters = 0

###################################### Training Loop ###################################

# REF: This training loop was inspired by the following PyTorch tutorial: 
# REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan.
# REF: The OASIS GAN demo 2 for Matthew Lockett 46988133, was also used heavily as inspiration.
print("Starting Training Loop...")
for epoch in range(hp.NUM_OF_EPOCHS):
   
   # Iterate over every batch in the dataset
   for i, batch in enumerate(train_loader, 0):
       
        # ---------------------------------------------------------------------------
        # (1) - Update the Discriminator by training with real images 
        # ---------------------------------------------------------------------------

        # Reset gradients of the discriminator
        disc.zero_grad()

        # Extract real images
        real_images = batch[0].to(device)

        # Create a real label to be applied to each image
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        # Forward pass the real batch through the discriminator
        output = disc(real_images).view(-1)

        # Calculate the real image discriminator loss
        real_loss_disc = criterion(output, label)

        # Calculate gradients of the real images for the discriminator
        real_loss_disc.backward()

        # Calculate average output value of discriminator
        real_disc_avg = output.mean().item()

        # ---------------------------------------------------------------------------
        # (2) - Update the Discriminator by training with fake images (by generator)
        # ---------------------------------------------------------------------------

        # Used to apply fake labels
        label.fill_(fake_label)

        # Generate a batch of latent vectors to input into the generator
        latent = torch.randn(batch_size, hp.LATENT_SIZE, device=device)

        # Generate fake images using the generator
        fake_images = gen(latent)

        # Apply the discriminator to classify fake images
        output = disc(fake_images.detach()).view(-1)

        # Calculate the fake image discriminator loss
        fake_loss_disc = criterion(output, label)

        # Calculate the gradients of the fake images for the discriminator
        fake_loss_disc.backward()

        # Calculate average output value of discriminator
        fake_disc_avg = output.mean().item()

        # Calculate the total error of the discriminator
        loss_disc = real_loss_disc + fake_loss_disc

        # Update the discriminator based on gradients
        disc_opt.step()

        # ---------------------------------------------------------------------------
        # (3) - Update the Generator 
        # ---------------------------------------------------------------------------

        # Reset gradients of the generator
        gen.zero_grad()

        # Apply a real label to fake image output, due to the generator wanting to appear 'real'
        label.fill_(real_label)

        # Output new classified images, due the discriminator being previously updated above
        output = disc(fake_images).view(-1)

        # Calculate the loss of the generator
        loss_gen = criterion(output, label)

        # Calculate the gradients of the generator
        loss_gen.backward()

        # Calculate average output value of discriminator due to generator "real" images
        gen_avg = output.mean().item()

        # Update the generator based on gradients
        gen_opt.step()

        # ---------------------------------------------------------------------------
        # (4) - Print out the statistics for each epoch 
        # ---------------------------------------------------------------------------

        # Output losses 
        if i % 15 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, hp.NUM_OF_EPOCHS, i, len(train_loader),
                    loss_disc.item(), loss_gen.item(), real_disc_avg, fake_disc_avg, gen_avg))

        # Save Losses for plotting later
        G_losses.append(loss_gen.item())
        D_losses.append(loss_disc.item())

        # Regularly save an image out of the generator to see progress overtime 
        if (iters % 239 == 0) or ((epoch == hp.NUM_OF_EPOCHS-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


# REF: The following lines of code for plotting losses and image outputs was inspired by 
# REF: the following PyTorch tutorial: 
# REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan.

# Output training loss plot
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator")
plt.plot(D_losses,label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(hp.SAVED_FIGURES_DIR, "training_loss_plot.png"), bbox_inches='tight', pad_inches=0)
plt.close()

# Visualise the evolution of the generator 
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save(os.path.join(hp.SAVED_FIGURES_DIR, "image_generation_over_time.gif"), writer='pillow', fps=1) 

# Grab a batch of real images from the train_loader
real_batch = next(iter(train_loader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig(os.path.join(hp.SAVED_FIGURES_DIR, "real_versus_fake_images.png"), bbox_inches='tight', pad_inches=0)
plt.close()