import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU...")


class UNet(nn.Module):

    def UNet2D_compact(input, latent_dim=64, activation=None, kernel=[3,3], channels=1, name_prefix=''):
        '''
        Uses tf.keras.Conv2D for the UNet (compact for small memory footprint)
        features 1 downsampling step & shortcut
        '''
        # Encoder
        net = layers.Norm_Conv2D(input, latent_dim//16, kernel, activation=ReLU()) # RF3
        down1 = layers.Norm_Conv2D(net, latent_dim//16, kernel, activation=ReLU()) # RF3
        net = MaxPool2D(2, padding='same')(down1) # down1

        net = layers.Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU()) # RF7
        down2 = layers.Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU()) # RF7
        net = MaxPool2D(2, padding='same')(down2) # down2

        net = layers.Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU()) # RF15
        down3 = layers.Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU()) # RF15
        net = MaxPool2D(2, padding='same')(down3) # down3

        net = layers.Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU()) # RF31
        down4 = layers.Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU()) # RF31
        net = MaxPool2D(2, padding='same')(down4) # down4

        net = layers.Norm_Conv2D(net, latent_dim, kernel, activation=ReLU()) # RF63
        latent = layers.Norm_Conv2D(net, latent_dim, kernel, activation=ReLU()) # RF63

        # Decoder
        up4 = layers.Norm_Conv2DTranspose(latent, latent_dim//2, kernel, 2, activation=ReLU()) # RF31
        net = Concatenate(axis=3)((up4,down4))
        net = layers.Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU()) # RF31
        net = layers.Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU()) # RF31

        up3 = layers.Norm_Conv2DTranspose(net, latent_dim//4, kernel, 2, activation=ReLU()) # RF15
        net = Concatenate(axis=3)((up3,down3))
        net = layers.Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU()) # RF15
        net = layers.Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU()) # RF15

        up2 = layers.Norm_Conv2DTranspose(net, latent_dim//8, kernel, 2, activation=ReLU()) # RF7
        net = Concatenate(axis=3)((up2,down2))
        net = layers.Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU()) # RF7
        net = layers.Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU()) # RF7

        up1 = layers.Norm_Conv2DTranspose(net, latent_dim//16, kernel, 2, activation=ReLU()) # RF3
        net = Concatenate(axis=3)((up1,down1))
        net = layers.Norm_Conv2D(net, latent_dim//16, kernel, activation=ReLU()) # RF3
        net = layers.Norm_Conv2D(net, latent_dim//16, kernel, activation=ReLU()) # RF3

        # Segmentation
        network=layers.Norm_Conv2D(net, channels, [1,1], activation=activation) # Output

        return network
