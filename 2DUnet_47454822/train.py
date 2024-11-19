import os

from keras.src.losses import Dice
from keras.src.optimizers import AdamW, Adam
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical

from dataset import load_data_2D, load_dir
from modules import unet_model
import numpy as np
from keras.src import ops, backend
import pathlib

# this is the loss function when we evaluate the model.
def dice_loss(y_true, y_pred, axis=None):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    inputs = y_true
    targets = y_pred

    intersection = ops.sum(inputs * targets, axis=axis)
    dice = ops.divide(
        2.0 * intersection,
        ops.sum(y_true, axis=axis)
        + ops.sum(y_pred, axis=axis)
        + backend.epsilon(),
    )

    return dice

# Load raw data
path = f"{pathlib.Path(__file__).parent.resolve()}/data/"

train_images = load_dir(f"{path}keras_slices_train/")
train_masks = load_dir(f"{path}keras_slices_seg_train/")

test_images = load_dir(f"{path}keras_slices_test/")
test_masks = load_dir(f"{path}keras_slices_seg_test/")


# Params
model_name = "full_depth_softmax_nodice_2epoc_train3"
epochs = 2


# Train the model if it doesn't already exist
if os.path.isfile(f'models/{model_name}.keras') is False:

    print('Training Model')

    # Process the data into the form it needs to be in to  train
    batch_size = 32
    train_images = np.expand_dims(train_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)

    train_masks = np.expand_dims(train_masks, axis=-1)
    train_masks = to_categorical(train_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes

    # Create the model then fit it to the data
    model = unet_model((256, 128, 1), batch_size=batch_size)
    model.fit(x=train_images, y=train_masks, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

    # Let the user know when its done and save the model
    print('Finished training, saving')
    model.save(f'models/{model_name}.keras')
    print('Saved and DONE')


# Test the model

# proccess the test data into the form the model needs
test_images = np.expand_dims(test_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)
test_masks = np.expand_dims(test_masks, axis=-1)
test_masks = to_categorical(test_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes

print("Testing Model")

# Load model then evaluate it
model = load_model(f'models/{model_name}.keras')
model.compile(optimizer=Adam(), loss=dice_loss, metrics=["accuracy"])
loss, accuracy = model.evaluate(test_images, test_masks)

# let the user know how it went
print(f"Dice Coefficient: {loss}")
print(f"Accuracy: {accuracy}")
