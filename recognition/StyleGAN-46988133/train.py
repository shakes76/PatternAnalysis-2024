"""
train.py created by Matthew Lockett 46988133
"""
import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import hyperparameters as hp
from dataset import load_ADNI_dataset
from modules import *

# Used to remove the Qt backend of matplotlib causing issues
matplotlib.use("Agg")

# Force the creation of a folder to save figures if not present 
os.makedirs(hp.SAVED_OUTPUT_DIR, exist_ok=True)

# PyTorch Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU.")

# Set random seed for reproducibility
print("Random Seed Used: ", hp.RANDOM_SEED)
random.seed(hp.RANDOM_SEED)
torch.manual_seed(hp.RANDOM_SEED)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Load the ADNI dataset images for training
train_loader = load_ADNI_dataset()

# Plot a sample of images from the ADNI dataset saved as 'adni_sample_images.png'
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Training Images for the ADNI Dataset")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, "adni_sample_images.png"), pad_inches=0)
plt.close()

# Setup both the generator and discriminator models
gen = Generator().to(device)
disc = Discriminator().to(device)

# Create the optimisers used by the genertator and discriminator during training 
gen_opt = optim.Adam(gen.parameters(), lr=hp.GEN_LEARNING_RATE, betas=(0.0, 0.99))
disc_opt = optim.Adam(disc.parameters(), lr=hp.DISC_LEARNING_RATE, betas=(0.0, 0.99))

# Implement a Cosine Annealing learning rate scheduler
# REF: Inspiration to use a scheduler came from code generated by ChatGPT-4o using the following prompt.
# REF: Prompt: What learning rate scheduler should I Choose for StyleGAN?
gen_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(gen_opt, T_0=math.ceil(hp.NUM_OF_EPOCHS*hp.COSINE_ANNEALING_RATE), T_mult=2)
disc_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(disc_opt, T_0=math.ceil(hp.NUM_OF_EPOCHS*hp.COSINE_ANNEALING_RATE), T_mult=2)

# Initialise Loss Functions
adversial_criterion = nn.BCELoss()
class_criterion = nn.CrossEntropyLoss()

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

# Start time of the training loop
training_start_time = time.time()

# REF: This training loop was inspired by the following PyTorch tutorial: 
# REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan.
# REF: The OASIS GAN demo 2 for Matthew Lockett 46988133, was also used heavily as inspiration.
# REF: The mixing regularisation portion was inspired by code generated by ChatGPT-4o.
# REF: Based on the following prompt: How can I implement mixing regularization into my StyleGAN model?
print("Starting Training Loop...")
for epoch in range(hp.NUM_OF_EPOCHS):
   
   # Start time of the epoch
   epoch_start_time = time.time()

   # Iterate over every batch in the dataset
   for i, (real_images, class_labels) in enumerate(train_loader, 0):
        
        # Move real images and labels from the ADNI dataset onto the GPU
        real_images = real_images.to(device)
        class_labels = class_labels.to(device)
        
        # Determine the batch size 
        batch_size = real_images.shape[0]
        
        # Create real, fake and random class labels
        real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
        rand_class_labels = torch.randint(0, 2, (batch_size,), device=device)
       
        # ---------------------------------------------------------------------------
        # (1) - Update the Discriminator by training with real images 
        # ---------------------------------------------------------------------------

        # Reset gradients of the discriminator
        disc.zero_grad()

        # Forward pass the real batch through the discriminator
        real_pred, class_real_pred, features_real = disc(real_images, labels=class_labels)

        # Calculate the real image discriminator losses
        real_loss_disc = adversial_criterion(real_pred.view(-1), real_labels)
        real_class_loss_disc = class_criterion(class_real_pred, class_labels)

        # Calculate average output value of discriminator
        real_disc_avg = real_pred.mean().item()

        # ---------------------------------------------------------------------------
        # (2) - Update the Discriminator by training with fake images (by generator)
        # ---------------------------------------------------------------------------

        # Generate a batch of latent vectors and labels to input into the generator
        latent1 = torch.randn(batch_size, hp.LATENT_SIZE, device=device)

        # Create a second latent space vector randomly for mixing regularisation
        if random.random() < hp.MIXING_PROB:
            latent2 = torch.randn(batch_size, hp.LATENT_SIZE, device=device)
        else:
            latent2 = None

        # Generate fake images using the generator and mixing regularisation
        fake_images = gen(latent1, latent2, mixing_ratio=random.uniform(0.5, 1.0), labels=rand_class_labels)

        # Apply the discriminator to classify fake images
        fake_pred, class_fake_pred, features_fake = disc(fake_images.detach(), labels=rand_class_labels)

        # Calculate the fake image discriminator losses
        fake_loss_disc = adversial_criterion(fake_pred.view(-1), fake_labels)
        fake_class_loss_disc = class_criterion(class_fake_pred, rand_class_labels)

        # Calculate average output value of discriminator
        fake_disc_avg = fake_pred.mean().item()

        # Calculate the total error of the discriminator
        tot_loss_disc = real_loss_disc + real_class_loss_disc + fake_loss_disc + fake_class_loss_disc

        # Update the discriminator based on gradients and losses
        tot_loss_disc.backward()
        disc_opt.step()

        # ---------------------------------------------------------------------------
        # (3) - Update the Generator 
        # ---------------------------------------------------------------------------

        # Reset gradients of the generator
        gen.zero_grad()

        # Output new classified images, due the discriminator being previously updated above
        fake_pred, class_fake_pred, features_fake = disc(fake_images, labels=rand_class_labels)

        # Calculate the losses of the generator
        loss_gen = adversial_criterion(fake_pred.view(-1), real_labels)
        loss_class_gen = class_criterion(class_fake_pred, rand_class_labels)

        # Calculate the total loss of the gradient
        tot_loss_gen = loss_gen + loss_class_gen

        # Calculate average output value of discriminator due to generator "real" images
        gen_avg = fake_pred.mean().item()

        # Update the generator based on gradients
        tot_loss_gen.backward()
        gen_opt.step()

        # ---------------------------------------------------------------------------
        # (4) - Print out the statistics for each epoch 
        # ---------------------------------------------------------------------------

        # Output losses 
        if i % 15 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, hp.NUM_OF_EPOCHS, i, len(train_loader),
                    tot_loss_disc.item(), tot_loss_gen.item(), real_disc_avg, fake_disc_avg, gen_avg))

        # Save Losses for plotting later
        G_losses.append(tot_loss_gen.item())
        D_losses.append(tot_loss_disc.item())

        # Regularly save an image out of the generator to see progress overtime 
        if (iters % 239 == 0) or ((epoch == hp.NUM_OF_EPOCHS-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

   # Step the schedulers after each epoch
   gen_scheduler.step()
   disc_scheduler.step()

   # Epoch training time
   epoch_duration = ((time.time() - epoch_start_time) / 60) # In minutes
   print(f"Epoch [{epoch + 1}/{hp.NUM_OF_EPOCHS}] completed in {epoch_duration:.2f} minutes")

# Total training time 
total_training_time = ((time.time() - training_start_time) / 60) # In minutes
print(f"Total training time: {total_training_time:.2f} minutes")

# Save the models for later use in inference 
torch.save(gen.state_dict(), os.path.join(hp.SAVED_OUTPUT_DIR, "generator_model.pth"))
torch.save(disc.state_dict(), os.path.join(hp.SAVED_OUTPUT_DIR, "discriminator_model.pth"))

###################################### Training Loop End #################################

# REF: The following lines of code for plotting losses and image outputs was inspired by 
# REF: the following PyTorch tutorial: 
# REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan.

# Output training loss plot
plt.figure(figsize=(10,5))
plt.title(f"Generator and Discriminator Loss During Training for {hp.NUM_OF_EPOCHS} Epochs")
plt.plot(G_losses,label="Generator")
plt.plot(D_losses,label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, "training_loss_plot.png"), pad_inches=0)
plt.close()

# Visualise the evolution of the generator 
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save(os.path.join(hp.SAVED_OUTPUT_DIR, "image_generation_over_time.gif"), writer='pillow', fps=1) 

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
plt.title(f"Generated Images - {hp.NUM_OF_EPOCHS} Epochs")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, "real_versus_fake_images.png"), pad_inches=0)
plt.close()
