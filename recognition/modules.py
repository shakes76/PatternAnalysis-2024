import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def init_unet(tensor_in, depth=64):
    """
    Function to initial the convolution layers of the UNet.
    parameters: 
    - tensor_in - the initial tensor filter to use for the model
    - depth - the number of channels/depth to use at the top layer of extraction (default 64)
    Return: 
    NULL
    """

    # For each layer of the UNET filter convolute twice and then pool.
    # Complete 2 convolutions to condense the image
    # Model based on https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing
    encode1 = layers.Conv2D(depth, 3, activation='relu', padding='same')(tensor_in)
    encode1 = layers.Conv2D(depth, 3, activation='relu', padding='same')(encode1)
    # Complete dimension reduction
    pool1 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(encode1)

    encode2 = layers.Conv2D(depth*2, 3, activation='relu', padding='same')(pool1)
    encode2 = layers.Conv2D(depth*2, 3, activation='relu', padding='same')(encode2)
    pool2 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(encode2)

    encode3 = layers.Conv2D(depth*4, 3, activation='relu', padding='same')(pool2)
    encode3 = layers.Conv2D(depth*4, 3, activation='relu', padding='same')(encode3)
    pool3 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(encode3)

    encode4 = layers.Conv2D(depth*8, 3, activation='relu', padding='same')(pool3)
    encode4 = layers.Conv2D(depth*8, 3, activation='relu', padding='same')(encode4)
    pool4 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(encode4)

    encode5 = layers.Conv2D(depth*16, 3, activation='relu', padding='same')(pool4)
    encode5 = layers.Conv2D(depth*16, 3, activation='relu', padding='same')(encode5)

    up1 = layers.Conv2DTranspose(depth*8, 2, strides=2, padding='same')(encode5)
    merge1 = layers.concatenate([encode4, up1])
    decode1 = layers.Conv2D(depth*8, 3, activation='relu', padding='same')(merge1)
    decode1 = layers.Conv2D(depth*8, 3, activation='relu', padding='same')(decode1)
    
    up2 = layers.Conv2DTranspose(depth*4, 2, strides=2, padding='same')(decode1)
    merge2 = layers.concatenate([encode3, up2])
    decode2 = layers.Conv2D(depth*4, 3, activation='relu', padding='same')(merge2)
    decode2 = layers.Conv2D(depth*4, 3, activation='relu', padding='same')(decode2)

    up3 = layers.Conv2DTranspose(depth*2, 2, strides=2, padding='same')(decode2)
    merge3 = layers.concatenate([encode2, up3])
    decode3 = layers.Conv2D(depth*2, 3, activation='relu', padding='same')(merge3)
    decode3 = layers.Conv2D(depth*2, 3, activation='relu', padding='same')(decode3)

    up4 = layers.Conv2DTranspose(depth, 2, strides=2, padding='same')(decode3)
    merge4 = layers.concatenate([encode1, up4])
    decode4 = layers.Conv2D(depth, 3, activation='relu', padding='same')(merge4)
    decode4 = layers.Conv2D(depth, 3, activation='relu', padding='same')(decode4)
    tensor_out = layers.Conv2D(1, 1, activation='softmax', padding='same')(decode4)
    model = models.Model(inputs=tensor_in, outputs=tensor_out)
    return model