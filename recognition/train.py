from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages
import modules

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Dice


# Set up the model from the modules file
unetModel = modules.unet()
unetModel.compile(optimizer = Adam(), loss = Dice(), metrics=['accuracy', Dice])

# Run the model and save 
trainResults = unetModel.fit(trainImages, trainSegImages, validation_data = (validateImages, validateSegImages), 
                        batch_size = 32, epochs=50)

# Evaluate the model using the test datasets
testResults = unetModel.evaluate(testImages, testSegImages)

print("Test loss: ", testResults[0])
print("Test accuracy: ", testResults[1])
print("Test dice: ", testResults[2])



