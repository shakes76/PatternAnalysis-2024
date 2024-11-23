import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore
from tensorflow.keras.metrics import BinaryAccuracy # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from modules import unet
from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages

# Function to calculate the dice ceofficient 
# Reference for Dice coefficient metric implementation:
# https://stackoverflow.com/questions/67018431/dice-coefficent-not-increasing-for-u-net-image-segmentation
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

# function to find the dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_metric(y_true, y_pred)

#Function to take the combined loss from binary cross entropy and dice 
def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + BinaryCrossentropy()(y_true, y_pred)

#learning rate scheduler and early stoppage
LearningRateScheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

#Create a unet instance 
unetModel = unet()

#compile the unet with a the adam optimiser and a learning rate of 0.0001. using the combined loss and the dice metric
unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=combined_loss, metrics=[dice_metric])

#print the unet model information
#unetModel.summary()

# Run the training on the model
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 4, epochs=12, callbacks=[earlyStopping, LearningRateScheduler], verbose=1)

# Run the trained model on the test datasets 
testResults = unetModel.evaluate(testImages, testSegImages, batch_size = 1)
unetModel.save('unet_model.keras')
