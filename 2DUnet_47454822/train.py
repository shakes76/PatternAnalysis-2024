import os

from keras.src.losses import Dice
from keras.src.optimizers import AdamW, Adam
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical
from dataset import load_data_2D, load_dir
from modules import unet_model
import numpy as np


path = "C:\\Users\\jjedm\\PatternAnalysis-2024\\HipMRI_study_keras_slices_data\\"

train_images = load_dir(f"{path}keras_slices_train\\")
train_masks = load_dir(f"{path}keras_slices_seg_train\\")

test_images = load_dir(f"{path}keras_slices_test\\")
test_masks = load_dir(f"{path}keras_slices_seg_test\\")
# train_images = test_images
# train_masks = test_masks


model_name = "full_depth_softmax_nodice_2epoc_train2"

# Train the model if it doesn't already exist
if os.path.isfile(f'models/{model_name}.keras') is False:
    # https://fdnieuwveldt.medium.com/building-advanced-custom-feature-transformation-pipelines-in-keras-using-easyflow-4c5fce545dc2

    print('Training Model')


    batch_size = 32
    train_images = np.expand_dims(train_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)
    # X = np.resize(X, (batch_size, 128, 128, 1))

    train_masks = np.expand_dims(train_masks, axis=-1)
    train_masks = to_categorical(train_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes
    # y = np.squeeze(y, axis=-1)
    # y = np.resize(y, (batch_size, 128, 128, 1))

    print(f"X Shape: {np.shape(train_images)}")
    print(f"y Shape: {np.shape(train_masks)}")


    model = unet_model((256, 128, 1), batch_size=batch_size)
    # model.compile(optimizer=AdamW(), loss=Dice(), metrics=['accuracy']) # used to be sparse_cat_crossent. find better loss

    model.fit(x=train_images, y=train_masks, batch_size=batch_size, epochs=2, shuffle=True, verbose=2)

    print('Finished training, saving')
    model.save(f'models/{model_name}.keras')
    print('Saved and DONE')


# Test the model

test_images = np.expand_dims(test_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)
test_masks = np.expand_dims(test_masks, axis=-1)
test_masks = to_categorical(test_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes

print("Testing Model")

model = load_model(f'models/{model_name}.keras')
model.compile(optimizer=Adam(), loss=Dice(), metrics=["accuracy"])
loss, accuracy = model.evaluate(test_images, test_masks)

print(f"Dice Coefficient: {1 - loss}")
print(f"Accuracy: {accuracy}")