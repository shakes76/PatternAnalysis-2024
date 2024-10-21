import tensorflow as tf
from modules import unet
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages

tf.keras.backend.clear_session()

# Reference for Dice coefficient metric implementation:
# https://stackoverflow.com/questions/67018431/dice-coefficent-not-increasing-for-u-net-image-segmentation
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

# Set up the model from the modules file
unetModel = unet()
#unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics=['accuracy'])

unetModel.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics = ['accuracy', dice_metric])

# Run the training on the model
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 2, epochs=1)

# Run the trained model on the test datasets 
testResults = unetModel.evaluate(testImages, testSegImages, batch_size = 32)
#trainPredictedSeg = unetModel.predict(testImages) Done in predict.py


