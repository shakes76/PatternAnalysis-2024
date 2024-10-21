import tensorflow as tf
tf.keras.backend.clear_session()

from modules import unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages


# Set up the model from the modules file
unetModel = unet()
unetModel.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Run the training on the model
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 2, epochs=1)

# Run the trained model on the test datasets 
testResults = unetModel.evaluate(testImages, testSegImages, batch_size = 32)
trainPredictedSeg = unetModel.predict(testImages)

# Extracting the loss, accuracy and dice score for the training and validation stats from the model
trainingLoss = trainResults.history["loss"]
trainingAccuracy = trainResults.history["accuracy"]
trainingValLoss = trainResults.history["val_loss"]
trainingValAccuracy = trainResults.history["val_accuracy"]

# Extracting the loss, accuract and dice score of the test set
testLoss = testResults[0]
testAccuracy = testResults[1]
