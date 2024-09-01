'''
A implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) designed for 256x256 greyscale images.
References:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://github.com/indiradutta/DC_GAN
https://arxiv.org/abs/1511.06434
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image

# Constants
IMAGE_SIZE = 256
LATENT_DIM = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
BETA1 = 0.5
NGPU = 1 # num GPUs


# Custom Dataset class for grayscale images
class GrayscaleImageDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_paths = []

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1] normalisation
        ])

        # Collect image paths
        image_folder = os.path.join(root_dir, f'keras_png_slices_{split}')
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.nii.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = NGPU
        self.main = nn.Sequential(
            # Input layer - Input size: LATENT_DIM x 1 x 1
            nn.ConvTranspose2d(in_channels=LATENT_DIM, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            
            # Upsampling layers - Input size: 1024 x 4 x 4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            # Input size: 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # Input size: 256 x 16 x 16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # Input size: 128 x 32 x 32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            
            # Output layer - Input size: 64 x 64 x 64
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output size: 1 x 128 x 128
        )
    
    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = NGPU
        self.main = nn.Sequential(
            # Input layer: Input size: 1 x 128 x 128
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Downsampling
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1028),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)