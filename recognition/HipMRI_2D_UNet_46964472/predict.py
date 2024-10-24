"""
Test script

@author Thuan Pham - 46964472
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from dataset import load_data_tf, load_data_predict
from utils import *
from matplotlib import pyplot as plt
import numpy as np

# Check if GPU available
print(tf.keras.__version__)
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("Using CPU")
else:
    print("Using GPU")

# hyper parameters
batch_size = 16
number_predict = 5
if len(sys.argv) < 2:
    print("Error: not enough argument.")
    print("Usage: python predict.py <dataset_dir> [number_predict] [batch_size].")
    print("Default for number_predict is 5 and batch_size is 16")
    quit()
else:
    dataset_dir = sys.argv[1]
    if len(sys.argv) > 2:
        number_predict = int(sys.argv[2])
    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])

# Load validate dataset
validate_image_dir = os.path.join(dataset_dir, "keras_slices_validate")
validate_seg_dir = os.path.join(dataset_dir, "keras_slices_seg_validate")
validate_dataset = load_data_tf(validate_image_dir, validate_seg_dir, batch_size=batch_size)

# Evaluate on all validate dataset
model = keras.models.load_model('model.keras', 
                                custom_objects={'dsc_loss': dsc_loss, 
                                                'dsc': dsc})
results = model.evaluate(validate_dataset)
print('Test Loss ', results[0] )
print('Test Dice Coefficients ', results[1] )

# Load random validate image and segmentation
images, images_norm, segs = load_data_predict(validate_image_dir, 
                                                 validate_seg_dir, 
                                                 number_predict)
# Predict
predict = model.predict(images_norm)
# Argmax to reverse one hot encoding
predict = np.argmax(predict, axis=3)

# Visualise results
plt.figure()
figure_pos = 0
for i in range(number_predict):    
    figure_pos += 1
    plt.subplot(3, number_predict, i + 1)
    plt.imshow(images[i])
    plt.title('Original Image')

    figure_pos += 1
    plt.subplot(3, number_predict, number_predict + i + 1)
    plt.imshow(segs[i])
    plt.title('Original Mask')

    figure_pos += 1
    plt.subplot(3, number_predict, number_predict * 2 + i + 1)
    plt.imshow(predict[i])
    plt.title('Prediction')
plt.show()