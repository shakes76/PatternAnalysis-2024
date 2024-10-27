import modules
import numpy as np
import dataset as data
import tensorflow as tf
from matplotlib import pyplot
from matplotlib import image

#Load only the test data
path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"
test_X = data.get_X_data(path, only_test=True)
test_Y = data.get_Y_data(path, only_test=True)

#Load model
model = modules.unet_model()
opt= tf.keras.optimizers.Adam(learning_rate = 0.005)
model.compile (optimizer=opt, loss= 'CategoricalCrossentropy' , metrics=[tf.keras.losses.Dice])
model.load_weights('UNet.weights.h5')
print(test_X.shape)
print(test_Y.shape)

#Evaluate Model against test data
print("Model evaluation:")
model.evaluate (test_X,test_Y)

#Generate predictions
predictions = np.round(model.predict(test_X))

#Plot predicted segment and true segments
im = 69 #which image to plot
for i in range (6):
  pyplot.title("Prediction_{}".format(i))
  pyplot.imshow(predictions[im,:,:,i])
  pyplot.savefig("graphs/Prediction_{}.png".format(i))
  pyplot.title("True_{}".format(i))
  pyplot.imshow(test_Y[im,:,:,i])
  pyplot.savefig("graphs/True_{}.png".format(i))

