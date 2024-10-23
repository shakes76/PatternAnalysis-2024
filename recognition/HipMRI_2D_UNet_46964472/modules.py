import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Dense, Flatten, MaxPool2D, Concatenate
from keras.initializers import GlorotNormal
from keras.models import Model
import numpy as np

def Norm_Conv2D(input, n_filters, 
                kernel_size=(3, 3), 
                strides=(1, 1), 
                activation=ReLU(), 
                use_bias=True, 
                kernel_initializer=GlorotNormal(), 
                **kwargs):
    conv_layer = Conv2D(n_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        activation=None,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        **kwargs)(input)
    norm_layer = BatchNormalization()(conv_layer)
    return activation(norm_layer)

def Norm_Conv2DTranspose(input, n_filters, 
                         kernel_size=(3, 3), 
                         strides=(1, 1), 
                         activation=ReLU(), 
                         use_bias=True, 
                         kernel_initializer=GlorotNormal(), 
                         **kwargs):
    conv_layer = Conv2DTranspose(n_filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 activation=None,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 **kwargs)(input)
    norm_layer = BatchNormalization()(conv_layer)
    return activation(norm_layer)

def UNet2D(input, latent_dim=64, activation=None, kernel=(3, 3), channels=1, name_prefix=''):
    # Encoder
    net = Norm_Conv2D(input, latent_dim//16, kernel, activation=ReLU())
    down1 = Norm_Conv2D(net, latent_dim//16, kernel, activation=ReLU())
    net = MaxPool2D(2, padding='same')(down1)
    net = Norm_Conv2D(input, latent_dim//8, kernel, activation=ReLU())
    down2 = Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU())
    net = MaxPool2D(2, padding='same')(down1)
    net = Norm_Conv2D(input, latent_dim//4, kernel, activation=ReLU())
    down3 = Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU())
    net = MaxPool2D(2, padding='same')(down1)
    net = Norm_Conv2D(input, latent_dim//2, kernel, activation=ReLU())
    down4 = Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU())
    net = MaxPool2D(2, padding='same')(down1)
    net = Norm_Conv2D(net, latent_dim, kernel, activation=ReLU())
    latent = Norm_Conv2D(net, latent_dim, kernel, activation=ReLU())

    # Decoder
    up4 = Norm_Conv2DTranspose(latent, latent_dim//2, kernel, activation=ReLU())
    net = Concatenate(axis=3)([up4, down4])
    net = Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU())
    net = Norm_Conv2D(net, latent_dim//2, kernel, activation=ReLU())
    up3 = Norm_Conv2DTranspose(net, latent_dim//4, kernel, 2, activation=ReLU())
    net = Concatenate(axis=3)([up3, down3])
    net = Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU())
    net = Norm_Conv2D(net, latent_dim//4, kernel, activation=ReLU())
    up2 = Norm_Conv2DTranspose(net, latent_dim//8, kernel, 2, activation=ReLU())
    net = Concatenate(axis=3)([up2, down2])
    net = Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU())
    net = Norm_Conv2D(net, latent_dim//8, kernel, activation=ReLU())
    up1 = Norm_Conv2DTranspose(net, latent_dim//16, kernel, 2, activation=ReLU())
    net = Concatenate(axis=3)([up1, down1])
    net = Norm_Conv2D(net, latent_dim//16, kernel, activation=ReLU())
    net = Norm_Conv2D(net, latent_dim//16, kernel, activation=ReLU())

    # Segmentation
    return Norm_Conv2D(net, channels, (1, 1), activation=activation)