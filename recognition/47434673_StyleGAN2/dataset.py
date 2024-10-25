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
from config import *

#############################################
# Data processing


def show_imgs(loader):
    '''
    Saves 5 images after the data transformation/augmentation and loading is complete and wrapped using dataloader.
    '''
    for i in range(5):
        features, _ = next(iter(loader))
        print(f"Feature batch shape: {features.size()}")
        img = features[0].squeeze()
        plt.imshow(img, cmap="gray")
        save_image(img*0.5+0.5, f"aug_img_{i}.png")

    # real_batch = next(iter(loader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()


def get_data(data, log_res, batchSize):
    '''
    Data Loader

    Resize: Resize images to a lower resolution as set in config, uses bicubic interpolation
    RandomVerticalFlip: Augment data by applying random vertical flips [probability=50%]
    ToTensor: Convert images to PyTorch Tensors
    Normalize: Normalise pixel value to have a mean and standard deviation of 0.5
    Grayscale: B&W images appropriate for the OASIS dataset
    '''
    # Create dataset
    dataset = dset.ImageFolder(root=data_root,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.BICUBIC),
                                  transforms.Grayscale(),
                                  #transforms.CenterCrop(image_size),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.Normalize(mean=[0.5], std=[0.5])]
                              ))
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)
    show_imgs(dataloader) # Display the training images

    return dataloader
