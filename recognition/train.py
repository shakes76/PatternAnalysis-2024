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
    """
    Function to calculate the max label dice coefficient

    """
    intersection = K.sum(y_true*y_predicted, axis = -1)
    union = K.sum(y_true, axis = -1) + K.sum(y_predicted, axis = -1)
    dice_coeff = (2*intersection) / (union)
    return dice_coeff


#Function to train the Unet model, using tensorflow's inbuilt keras.model.fit() function
def train_unet(data_train, data_train_seg, data_validate, data_validate_seg, epochs=32, batch_size=8):
    """
    Training function for the unet based on provided data and segmented data.
    """
    init_inputs = layers.Input(shape=(256, 128, 1))
    model = init_unet(init_inputs)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
    # Compile and train the UNET model on the training set and validation set
    # Using the loss function for ones hot encoding
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', dice_coeff])
    model.fit(data_train, data_train_seg, epochs=epochs, batch_size=batch_size, validation_data=(data_validate, data_validate_seg), callbacks=[early_stopping], verbose=1)
    # save model reference: https://www.tensorflow.org/tutorials/keras/save_and_load
    model.save('mri_unet.keras')

def main():
    data_train = load_data_2D()
    data_train_seg = load_data_2D()

    data_validate = load_data_2D()
    data_validate_seg = load_data_2D()

    train_unet(data_train, data_train_seg, data_validate, data_validate_seg)

if __name__ == "__main__":
    main()