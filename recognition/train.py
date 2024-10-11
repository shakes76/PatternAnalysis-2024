from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages
from modules import unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Dice

# Set up the model from the modules file
unetModel = unet()
unetModel.compile(optimizer = Adam(), loss = Dice(), metrics=['accuracy', Dice])

# Run the training on the model
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 32, epochs=1)

# Run the trained model on the test datasets 
testResults = unetModel.evaluate(testImages, testSegImages, batch_size = 32)

# Extracting the loss, accuracy and dice score for the training and validation stats from the model
trainingLoss = trainResults.history["loss"]
trainingAccuracy = trainResults.history["accuracy"]
trainingDiceScore = trainResults.history["dice_coefficient"]
trainingValLoss = trainResults.history["val_loss"]
trainingValAccuracy = trainResults.history["val_accuracy"]
trainingValDiceScore = trainResults.history["val_dice_coefficient"]

# Extracting the loss, accuract and dice score of the test set
testLoss = testResults[0]
testAccuracy = testResults[1]
testDiceScore = testResults[2]





