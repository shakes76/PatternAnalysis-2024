# modules.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build the 3D U-Net model
def unet_3d(input_shape, num_classes):
    """
    This is currently a standard implementation of the U Net.
    REF: This code came from the tutorial Youtube video. https://youtu.be/GAYJ81M58y8?si=jQqSwcT5bDk3JE-5
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    # Does 3D convolution than a max pooling
    c1 = layers.Conv3D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D(2)(c1)

    c2 = layers.Conv3D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D(2)(c2)

    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling3D(2)(c3)

    c4 = layers.Conv3D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling3D(2)(c4)

    # Bottleneck
    bn = layers.Conv3D(256, 3, activation='relu', padding='same')(p4)
    bn = layers.Conv3D(256, 3, activation='relu', padding='same')(bn)

    # Decoder
    # The transpose does the up convolution.
    u1 = layers.Conv3DTranspose(128, 2, strides=2, padding='same')(bn)
    # Connecting to the layers from the encoder.
    u1 = layers.concatenate([u1, c4])
    c5 = layers.Conv3D(128, 3, activation='relu', padding='same')(u1)
    c5 = layers.Conv3D(128, 3, activation='relu', padding='same')(c5)

    u2 = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(c5)
    u2 = layers.concatenate([u2, c3])
    c6 = layers.Conv3D(64, 3, activation='relu', padding='same')(u2)
    c6 = layers.Conv3D(64, 3, activation='relu', padding='same')(c6)

    u3 = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(c6)
    u3 = layers.concatenate([u3, c2])
    c7 = layers.Conv3D(32, 3, activation='relu', padding='same')(u3)
    c7 = layers.Conv3D(32, 3, activation='relu', padding='same')(c7)

    u4 = layers.Conv3DTranspose(16, 2, strides=2, padding='same')(c7)
    u4 = layers.concatenate([u4, c1])
    c8 = layers.Conv3D(16, 3, activation='relu', padding='same')(u4)
    c8 = layers.Conv3D(16, 3, activation='relu', padding='same')(c8)

    outputs = layers.Conv3D(num_classes, 1, activation='softmax')(c8)

    model = keras.Model(inputs, outputs)
    return model

# Loss functions
def dice_loss(y_true, y_pred):
    """
    The dice loss function is given by calculating the overlap of to areas, and
    deviding the union of these two areas.
    """
    # tf.reduce_sum and tf.reduce_mean is for handling the entire batch rather than for a single data.
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3,4))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3,4))
    return 1 - tf.reduce_mean(numerator / (denominator + 1e-6))

def combined_loss(y_true, y_pred):
    """
    This is the combined loss of the dice score and Cross-Entropy Loss to prevent overfitting.
    Without this combined loss the dice score is only around 5.2 but with this the score can go up to 8.0.
    """
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return ce_loss + dl

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    This calculates the dice coefficient.
    """
    # Flatten is really important without it, the code will give an error.
    # Flatten converts a 3D array into a 1 D array.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
