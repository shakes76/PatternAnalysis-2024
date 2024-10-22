# modules.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

def unet_3d(input_shape, num_classes):
    """
    Improved U-Net model with Instance Normalization and Leaky ReLU activation functions.
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    # Block 1
    c1 = layers.Conv3D(16, 3, padding='same', use_bias=False)(inputs)
    c1 = tfa.layers.InstanceNormalization()(c1)
    c1 = layers.LeakyReLU(alpha=0.01)(c1)
    c1 = layers.Conv3D(16, 3, padding='same', use_bias=False)(c1)
    c1 = tfa.layers.InstanceNormalization()(c1)
    c1 = layers.LeakyReLU(alpha=0.01)(c1)
    p1 = layers.MaxPooling3D(2)(c1)

    # Block 2
    c2 = layers.Conv3D(32, 3, padding='same', use_bias=False)(p1)
    c2 = tfa.layers.InstanceNormalization()(c2)
    c2 = layers.LeakyReLU(alpha=0.01)(c2)
    c2 = layers.Conv3D(32, 3, padding='same', use_bias=False)(c2)
    c2 = tfa.layers.InstanceNormalization()(c2)
    c2 = layers.LeakyReLU(alpha=0.01)(c2)
    p2 = layers.MaxPooling3D(2)(c2)

    # Block 3
    c3 = layers.Conv3D(64, 3, padding='same', use_bias=False)(p2)
    c3 = tfa.layers.InstanceNormalization()(c3)
    c3 = layers.LeakyReLU(alpha=0.01)(c3)
    c3 = layers.Conv3D(64, 3, padding='same', use_bias=False)(c3)
    c3 = tfa.layers.InstanceNormalization()(c3)
    c3 = layers.LeakyReLU(alpha=0.01)(c3)
    p3 = layers.MaxPooling3D(2)(c3)

    # Block 4
    c4 = layers.Conv3D(128, 3, padding='same', use_bias=False)(p3)
    c4 = tfa.layers.InstanceNormalization()(c4)
    c4 = layers.LeakyReLU(alpha=0.01)(c4)
    c4 = layers.Conv3D(128, 3, padding='same', use_bias=False)(c4)
    c4 = tfa.layers.InstanceNormalization()(c4)
    c4 = layers.LeakyReLU(alpha=0.01)(c4)
    p4 = layers.MaxPooling3D(2)(c4)

    # Bottleneck
    bn = layers.Conv3D(256, 3, padding='same', use_bias=False)(p4)
    bn = tfa.layers.InstanceNormalization()(bn)
    bn = layers.LeakyReLU(alpha=0.01)(bn)
    bn = layers.Conv3D(256, 3, padding='same', use_bias=False)(bn)
    bn = tfa.layers.InstanceNormalization()(bn)
    bn = layers.LeakyReLU(alpha=0.01)(bn)

    # Decoder
    # Up Block 1
    u1 = layers.Conv3DTranspose(128, 2, strides=2, padding='same', use_bias=False)(bn)
    u1 = tfa.layers.InstanceNormalization()(u1)
    u1 = layers.LeakyReLU(alpha=0.01)(u1)
    u1 = layers.concatenate([u1, c4])
    c5 = layers.Conv3D(128, 3, padding='same', use_bias=False)(u1)
    c5 = tfa.layers.InstanceNormalization()(c5)
    c5 = layers.LeakyReLU(alpha=0.01)(c5)
    c5 = layers.Conv3D(128, 3, padding='same', use_bias=False)(c5)
    c5 = tfa.layers.InstanceNormalization()(c5)
    c5 = layers.LeakyReLU(alpha=0.01)(c5)

    # Up Block 2
    u2 = layers.Conv3DTranspose(64, 2, strides=2, padding='same', use_bias=False)(c5)
    u2 = tfa.layers.InstanceNormalization()(u2)
    u2 = layers.LeakyReLU(alpha=0.01)(u2)
    u2 = layers.concatenate([u2, c3])
    c6 = layers.Conv3D(64, 3, padding='same', use_bias=False)(u2)
    c6 = tfa.layers.InstanceNormalization()(c6)
    c6 = layers.LeakyReLU(alpha=0.01)(c6)
    c6 = layers.Conv3D(64, 3, padding='same', use_bias=False)(c6)
    c6 = tfa.layers.InstanceNormalization()(c6)
    c6 = layers.LeakyReLU(alpha=0.01)(c6)

    # Up Block 3
    u3 = layers.Conv3DTranspose(32, 2, strides=2, padding='same', use_bias=False)(c6)
    u3 = tfa.layers.InstanceNormalization()(u3)
    u3 = layers.LeakyReLU(alpha=0.01)(u3)
    u3 = layers.concatenate([u3, c2])
    c7 = layers.Conv3D(32, 3, padding='same', use_bias=False)(u3)
    c7 = tfa.layers.InstanceNormalization()(c7)
    c7 = layers.LeakyReLU(alpha=0.01)(c7)
    c7 = layers.Conv3D(32, 3, padding='same', use_bias=False)(c7)
    c7 = tfa.layers.InstanceNormalization()(c7)
    c7 = layers.LeakyReLU(alpha=0.01)(c7)

    # Up Block 4
    u4 = layers.Conv3DTranspose(16, 2, strides=2, padding='same', use_bias=False)(c7)
    u4 = tfa.layers.InstanceNormalization()(u4)
    u4 = layers.LeakyReLU(alpha=0.01)(u4)
    u4 = layers.concatenate([u4, c1])
    c8 = layers.Conv3D(16, 3, padding='same', use_bias=False)(u4)
    c8 = tfa.layers.InstanceNormalization()(c8)
    c8 = layers.LeakyReLU(alpha=0.01)(c8)
    c8 = layers.Conv3D(16, 3, padding='same', use_bias=False)(c8)
    c8 = tfa.layers.InstanceNormalization()(c8)
    c8 = layers.LeakyReLU(alpha=0.01)(c8)

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
