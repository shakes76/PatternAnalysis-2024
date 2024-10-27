# This is for the training algorithm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from modules import init_unet
from dataset import load_data_2D, to_channels

# https://medium.com/@vipul.sarode007/u-net-unleashed-a-step-by-step-guide-on-implementing-and-training-your-own-segmentation-model-in-a90ed89399c6
def dice_coeff(y_true, y_pred, smooth = 1):
    intersection = K.sum(y_true*y_pred, axis = -1)
    union = K.sum(y_true, axis = -1) + K.sum(y_pred, axis = -1)
    dice_coeff = (2*intersection+smooth) / (union + smooth)
    return dice_coeff

def train_unet( data_train, data_train_cat, data_test, data_test_cat, epochs=32, batch_size=8):
    init_inputs = layers.Input(shape=(256, 128, 1))
    model = init_unet(init_inputs)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', dice_coeff])
    model.fit(data_train, data_train_cat, epochs=epochs, batch_size=batch_size, validation_data=(data_test, data_test_cat), callbacks=[early_stopping])