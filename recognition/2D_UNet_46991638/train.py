"""
This file contains the main test script for training the model
"""
import dataset as data
import modules
import tensorflow as tf
from matplotlib import pyplot
import numpy as np

l_rate = 0.0001
epochs = 50


path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"
train_X, validate_X, test_X = data.get_X_data(path)
train_Y, validate_Y, test_Y = data.get_Y_data(path)

#Initialise model and data generators        
model = modules.unet_model()
train_gen = data.DataGenerator(train_X, train_Y, batch_size=32)
val_gen = data.DataGenerator(validate_X, validate_Y, batch_size=32)

opt= tf.keras.optimizers.Adam(learning_rate = l_rate)
model.compile (optimizer=opt, loss= 'CategoricalCrossentropy' , metrics=[tf.keras.losses.Dice])
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

print("Model evaluation:")
model.evaluate(test_X,test_Y)

model.save_weights('UNet.weights.h5')

#Plot loss function
pyplot.title('Dice Similarity Coefficient')
pyplot.plot([1 - x for x in history.history['dice']], color='blue', label='train')
pyplot.plot([1 - x for x in history.history['val_dice']], color='orange', label='test')
pyplot.legend(('training','validation'))
pyplot.savefig("Dice_Coeff")
pyplot.clf()
