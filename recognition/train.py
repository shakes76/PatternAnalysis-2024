import tensorflow as tf
from modules import unet
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages
import numpy as np

# Reference for Dice coefficient metric implementation:
# https://stackoverflow.com/questions/67018431/dice-coefficent-not-increasing-for-u-net-image-segmentation
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

def dice_loss(y_true, y_pred):
    return 1 - dice_metric(y_true, y_pred)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

# Set up the model from the modules file
unetModel = unet()
unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics = ['accuracy',dice_metric])
#unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=dice_loss, metrics=['accuracy', dice_metric])

#unetModel.summary()

# Run the training on the model
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 8, epochs=20, callbacks=[early_stopping])

# Run the trained model on the test datasets 
testResults = unetModel.evaluate(testImages, testSegImages, batch_size = 32)
unetModel.save('unet_model.keras')