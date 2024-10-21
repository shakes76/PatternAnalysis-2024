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


# Reference for Dice coefficient metric implementation:
# https://stackoverflow.com/questions/67018431/dice-coefficent-not-increasing-for-u-net-image-segmentation
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

def dice_loss(y_true, y_pred):
    return 1 - dice_metric(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + BinaryCrossentropy()(y_true, y_pred)

learn_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

# Set up the model from the modules file
unetModel = unet()
unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics = ['accuracy',dice_metric])
#unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=dice_loss, metrics=[dice_metric])

#unetModel.summary()

# Run the training on the model
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 2, epochs=3, callbacks=[early_stopping, learn_rate_scheduler], verbose=1)

# Run the trained model on the test datasets 
testResults = unetModel.evaluate(testImages, testSegImages, batch_size = 1)
unetModel.save('unet_model.keras')