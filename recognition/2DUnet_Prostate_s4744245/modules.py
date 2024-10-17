import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(n_classes, input_size=(256, 128, 1)):
    inputs = layers.Input(input_size)
    
    # Encoder (downsampling)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)    # 256x128x64
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)     # 256x128x64
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)                       # 128x64x64

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)    # 128x64x128
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)    # 128x64x128
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)                       # 64x32x128

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)    # 64x32x256
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)    # 64x32x256
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)                       # 32x16x256

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)    # 32x16x512
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)    # 32x16x512
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)                       # 16x8x512

    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)   # 16x8x1024
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)   # 16x8x1024

    # Decoder (upsampling)
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5) # 32x16x512
    merge6 = layers.concatenate([conv4, up6], axis=3)                          # Skip connection with conv4
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)   # 32x16x512
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)    # 32x16x512

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6) # 64x32x256
    merge7 = layers.concatenate([conv3, up7], axis=3)                          # Skip connection with conv3
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)   # 64x32x256
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)    # 64x32x256

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7) # 128x64x128
    merge8 = layers.concatenate([conv2, up8], axis=3)                          # Skip connection with conv2
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)   # 128x64x128
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)    # 128x64x128

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)  # 256x128x64
    merge9 = layers.concatenate([conv1, up9], axis=3)                          # Skip connection with conv1
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)    # 256x128x64
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)     # 256x128x64

    conv10 = layers.Conv2D(n_classes, (1, 1), activation='softmax')(conv9)                  # 256x128xn_classes - Output segmentation map

    model = models.Model(inputs=inputs, outputs=conv10)
    
    return model

