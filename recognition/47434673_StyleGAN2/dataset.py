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
