import dataset as data
import modules
import tensorflow as tf
from matplotlib import pyplot
import numpy as np

l_rate = 0.005
epochs = 30

X_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_"
train_X = data.load(X_path + "train")
validate_X = data.load(X_path + "validate")
test_X = data.load(X_path + "test")

train_X = data.process_data(train_X)
validate_X = data.process_data(validate_X)
test_X = data.process_data(test_X)

seg_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_"
train_Y = data.load(seg_path + "train", label=True)

validate_Y = data.load(seg_path + "validate", label=True)

test_Y = data.load(seg_path + "test", label=True)
print("X shapes")
print(train_X.shape)
print(validate_X.shape)
print(test_X.shape)
print("X is :")
print(np.amin(train_X))
print(np.amax(train_X))
print(np.amin(test_X))
print(np.amax(test_X))
print(train_X[1,1:5,1:5])
print("Y is")
print(np.amin(train_Y))
print(np.amax(train_Y))
print(np.amin(test_Y))
print(np.amax(test_Y))
print("Labeled shapes")
print(train_Y.shape)
print(validate_Y.shape)
print(test_Y.shape)

        
model = modules.unet_model()
train_gen = data.DataGenerator(train_X, train_Y, batch_size=32)
val_gen = data.DataGenerator(validate_X, validate_Y, batch_size=32)

opt= tf.keras.optimizers.Adam(learning_rate = l_rate)
model.compile (optimizer=opt, loss= 'CategoricalCrossentropy' , metrics=[tf.keras.losses.Dice])
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
#history = model.fit(train_X, train_Y,  validation_data=(validate_X, validate_Y), batch_size=20,shuffle='True',epochs=epochs)

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
