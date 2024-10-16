from keras.src.losses import Dice
from keras.src.optimizers import AdamW, Adam
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical
from dataset import load_data_2D, load_dir
from modules import unet_model
import numpy as np


path = "C:\\Users\\jjedm\\PatternAnalysis-2024\\HipMRI_study_keras_slices_data\\"

test_images = load_dir(f"{path}keras_slices_test\\")
test_masks = load_dir(f"{path}keras_slices_seg_test\\")


model_name = "full_depth_softmax_nodice_2epoc_train2"
test_images = np.expand_dims(test_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)

test_masks = np.expand_dims(test_masks, axis=-1)
test_masks = to_categorical(test_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes

print("Loading Model")
model = load_model(f'models/{model_name}.keras')

model.compile(optimizer=Adam(), loss=Dice(), metrics=["accuracy"])

loss, accuracy = model.evaluate(test_images, test_masks)

print(f"Dice Coefficient: {1 - loss}")
print(f"Accuracy: {accuracy}")