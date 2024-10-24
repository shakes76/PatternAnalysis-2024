import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPool2D, Concatenate
from keras.initializers import GlorotNormal
from keras import Model

def Norm_Conv2D(input, n_filters, 
                kernel_size=(3, 3), 
                strides=(1, 1), 
                activation="relu", 
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
    return Activation(activation)(norm_layer)

def Norm_Conv2DTranspose(input, n_filters, 
                         kernel_size=(3, 3), 
                         strides=(1, 1), 
                         activation="relu", 
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
    return Activation(activation)(norm_layer)

def UNet2D(input_size, latent_dim=64, activation=None, kernel=(3, 3), channels=1, name_prefix=''):
    input = Input(input_size)

    # Encoder
    net = Norm_Conv2D(input, latent_dim//16, kernel)
    down1 = Norm_Conv2D(net, latent_dim//16, kernel)
    net = MaxPool2D(2, padding='same')(down1)
    net = Norm_Conv2D(net, latent_dim//8, kernel)
    down2 = Norm_Conv2D(net, latent_dim//8, kernel)
    net = MaxPool2D(2, padding='same')(down2)
    net = Norm_Conv2D(net, latent_dim//4, kernel)
    down3 = Norm_Conv2D(net, latent_dim//4, kernel)
    net = MaxPool2D(2, padding='same')(down3)
    net = Norm_Conv2D(net, latent_dim//2, kernel)
    down4 = Norm_Conv2D(net, latent_dim//2, kernel)
    net = MaxPool2D(2, padding='same')(down4)
    net = Norm_Conv2D(net, latent_dim, kernel)
    latent = Norm_Conv2D(net, latent_dim, kernel)

    # Decoder
    up4 = Norm_Conv2DTranspose(latent, latent_dim//2, kernel, 2)
    net = Concatenate(axis=3)([up4, down4])
    net = Norm_Conv2D(net, latent_dim//2, kernel)
    net = Norm_Conv2D(net, latent_dim//2, kernel)
    up3 = Norm_Conv2DTranspose(net, latent_dim//4, kernel, 2)
    net = Concatenate(axis=3)([up3, down3])
    net = Norm_Conv2D(net, latent_dim//4, kernel)
    net = Norm_Conv2D(net, latent_dim//4, kernel)
    up2 = Norm_Conv2DTranspose(net, latent_dim//8, kernel, 2)
    net = Concatenate(axis=3)([up2, down2])
    net = Norm_Conv2D(net, latent_dim//8, kernel)
    net = Norm_Conv2D(net, latent_dim//8, kernel)
    up1 = Norm_Conv2DTranspose(net, latent_dim//16, kernel, 2)
    net = Concatenate(axis=3)([up1, down1])
    net = Norm_Conv2D(net, latent_dim//16, kernel)
    net = Norm_Conv2D(net, latent_dim//16, kernel)

    # Segmentation
    output = Norm_Conv2D(net, channels, (1, 1), activation=activation)
    return Model(input, output)