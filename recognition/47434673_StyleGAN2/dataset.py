from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython.display import HTML
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from math import sqrt
from tqdm import tqdm

import modules
import utils
import predict
import train




# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


DATA = "C:\Users\kylie\OneDrive\Documents\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_train"




#############################################
# Data processing

'''
Saves 5 images after the data transformation/augmentation and loading is complete and wrapped using dataloader.
'''
def show_imgs(loader):
    # for i in range(5):
    #     features, _ = next(iter(loader))
    #     print(f"Feature batch shape: {features.size()}")
    #     img = features[0].squeeze()
    #     plt.imshow(img, cmap="gray")
    #     save_image(img*0.5+0.5, f"aug_img_{i}.png")

    real_batch = next(iter(loader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

'''
Data Loader

Resize: Resize images to a lower resolution as set in config, uses bicubic interpolation
RandomHorizontalFlip: Augment data by applying random horizontal flips [probability=50%]
ToTensor: Convert images to PyTorch Tensors
Normalize: Normalize pixel value to have a mean and standard deviation of 0.5
'''
def get_data(data, log_res, batchSize):
    # Create the dataset
    dataset = dset.ImageFolder(root=DATA,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize((utils.image_height, utils.image_width), interpolation=transforms.InterpolationMode.BICUBIC),
                                  transforms.Grayscale(),
                                  #transforms.CenterCrop(image_size),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.Normalize(mean=[0.5], std=[0.5])]
                              ))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=utils.batch_size,
                                            shuffle=True)
    show_imgs(dataloader)
        
    return dataloader






####################################################
# Main function

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# Create batch of latent vectors used to visualise the progression of the generator
fixed_noise = torch.randn(64, utils.nz, 1, 1, device=device)

# Dataloader
loader = get_data(DATA, utils.log_resolution, utils.batch_size)

gen = modules.Generator(utils.log_resolution, utils.w_dim).to(device)
critic = modules.Discriminator(utils.log_resolution).to(device)
mapping_network = modules.MappingNetwork(utils.z_dim, utils.w_dim).to(device)
path_length_penalty = modules.PathLengthPenalty(0.99).to(device)

# Initialise Adam optimiser
opt_gen = optim.Adam(gen.parameters(), lr=utils.learning_rate, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=utils.learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=utils.learning_rate, betas=(0.0, 0.99))


# # Check if there are pre-trained models to load
load_model = False  # Set to True if you want to load a pre-trained model
if load_model:
    gen.load_state_dict(torch.load('/content/drive/My Drive/COMP3710/assignment-two/netG.pth'))
    critic.load_state_dict(torch.load('/content/drive/My Drive/COMP3710/assignment-two/netD.pth'))
    print("Loaded pre-trained models.")

if not load_model:
    # Train the following modules
    gen.train()
    critic.train()
    mapping_network.train()

# Keeps a Log of total loss over the training
G_Loss = []
D_Loss = []
img_list = []

# loop over total epcoh.
for epoch in range(utils.epochs):
    curr_Gloss, curr_Dloss = train.train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )

    # Append the current loss to the main list
    G_Loss.extend(curr_Gloss)
    D_Loss.extend(curr_Dloss)

    # Save generator's fake image on every 50th epoch
    if epoch % 10 == 0:
        predict.generate_examples(gen, mapping_network, epoch, device)

    if (epoch % 10 == 0) or (epoch == utils.epoch-1): #and (i == len(loader)-1)):
        fake = gen(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        

        # Save the models after training
torch.save(gen.state_dict(), '/content/drive/My Drive/COMP3710/assignment-two/netG.pth')
torch.save(critic.state_dict(), '/content/drive/My Drive/COMP3710/assignment-two/netD.pth')

predict.plot_loss(G_Loss, D_Loss)

# Plot the Generator and Discriminator losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_Loss, label="G")
plt.plot(D_Loss, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


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




