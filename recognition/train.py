from dataset import testImages, trainImages, validateImages, testSegImages, trainSegImages, validateSegImages
import modules

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Set up the model from the modules file
unetModel = modules.unet()
unetModel.compile(optimizer = Adam(), loss = BinaryCrossentropy(), metrics=['accuracy'])
unetModel.fit(trainImages, trainSegImages, epochs=3)