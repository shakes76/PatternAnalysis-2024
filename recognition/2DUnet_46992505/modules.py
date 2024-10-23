import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np


# Dice coefficient function
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # Flatten the arrays
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)  # Flatten the arrays
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Negative Dice coefficient (used as a loss function)
def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def unet_model(input_size=(256, 128, 1)):
    inputs = layers.Input(input_size)

    #Encoder 
    conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
    bn1 = layers.Activation("relu")(conv1)
    conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn1)
    bn1 = layers.BatchNormalization(axis=3)(conv1)
    bn1 = layers.Activation("relu")(bn1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(pool1)
    bn2 = layers.Activation("relu")(conv2)
    conv2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn2)
    bn2 = layers.BatchNormalization(axis=3)(conv2)
    bn2 = layers.Activation("relu")(bn2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(pool2)
    bn3 = layers.Activation("relu")(conv3)
    conv3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn3)
    bn3 = layers.BatchNormalization(axis=3)(conv3)
    bn3 = layers.Activation("relu")(bn3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(pool3)
    bn4 = layers.Activation("relu")(conv4)
    conv4 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn4)
    bn4 = layers.BatchNormalization(axis=3)(conv4)
    bn4 = layers.Activation("relu")(bn4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    #Bottleneck
    conv5 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(pool4)
    bn5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(bn5)
    bn5 = layers.BatchNormalization(axis=3)(conv5)
    bn5 = layers.Activation("relu")(bn5)

    #Decoder
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    #Merge variables are for implementing the skip connections
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(merge6)
    bn6 = layers.Activation("relu")(conv6)
    conv6 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn6)
    bn6 = layers.BatchNormalization(axis=3)(conv6)
    bn6 = layers.Activation("relu")(bn6)



    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(merge7)
    bn7 = layers.Activation("relu")(conv7)
    conv7 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn7)
    bn7 = layers.BatchNormalization(axis=3)(conv7)
    bn7 = layers.Activation("relu")(bn7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(merge8)
    bn8 = layers.Activation("relu")(conv8)
    conv8 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn8)
    bn8 = layers.BatchNormalization(axis=3)(conv8)
    bn8 = layers.Activation("relu")(bn8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(merge9)
    bn9 = layers.Activation("relu")(conv9)
    conv9 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn9)
    bn9 = layers.BatchNormalization(axis=3)(conv9)
    bn9 = layers.Activation("relu")(bn9)
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs, conv10)
    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=dice_loss, metrics=["accuracy"])

    return model

