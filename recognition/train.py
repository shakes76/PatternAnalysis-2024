# This is for the training algorithm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from modules import init_unet
from dataset import load_data_2D, to_channels

# https://medium.com/@vipul.sarode007/u-net-unleashed-a-step-by-step-guide-on-implementing-and-training-your-own-segmentation-model-in-a90ed89399c6
# Function to calculate the mathmatical dice coefficient
def dice_coeff(y_true, y_predicted):
    intersection = K.sum(y_true*y_predicted, axis = -1)
    union = K.sum(y_true, axis = -1) + K.sum(y_predicted, axis = -1)
    dice_coeff = (2*intersection) / (union)
    return dice_coeff


#Function to train the Unet model, using tensorflow's inbuilt keras.model.fit() function
def train_unet( data_train, data_train_seg, data_validate, data_validate_seg, epochs=32, batch_size=8):
    init_inputs = layers.Input(shape=(256, 128, 1))
    model = init_unet(init_inputs)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
    # Compile and train the UNET model on the training set and validation set
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', dice_coeff])
    model.fit(data_train, data_train_seg, epochs=epochs, batch_size=batch_size, validation_data=(data_validate, data_validate_seg), callbacks=[early_stopping])
    return model
    
