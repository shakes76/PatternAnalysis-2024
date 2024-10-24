import tensorflow as tf
from tensorflow import keras
import os
from dataset import *
from modules import UNet2D
from utils import *
from matplotlib import pyplot as plt
import numpy as np

print(tf.keras.__version__)
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("Using CPU")
else:
    print("Using GPU")

image_height = 256
image_width = 128
batch_size = 16
channels = 6

validate_image_dir = os.path.join("keras_slices_data", "keras_slices_validate")
validate_seg_dir = os.path.join("keras_slices_data", "keras_slices_seg_validate")
validate_dataset = load_data_tf(validate_image_dir, validate_seg_dir, batch_size=batch_size)

model = keras.models.load_model('model.keras', 
                                custom_objects={'dsc_loss': dsc_loss, 
                                                'dsc': dsc})
model.summary()

results = model.evaluate(validate_dataset)

print('Test Loss ', results[0] )
print('Test Dice Coefficients ', results[1] )