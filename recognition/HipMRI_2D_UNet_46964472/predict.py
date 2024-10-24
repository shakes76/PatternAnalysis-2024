"""
Test script

@author Thuan Pham - 46964472
"""
import os
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
number_predict = 10

# Load validate dataset
validate_image_dir = os.path.join("keras_slices_data", "keras_slices_validate")
validate_seg_dir = os.path.join("keras_slices_data", "keras_slices_seg_validate")
validate_dataset = load_data_tf(validate_image_dir, validate_seg_dir, batch_size=batch_size)

# Evaluate on all validate dataset
model = keras.models.load_model('model.keras', 
                                custom_objects={'dsc_loss': dsc_loss, 
                                                'dsc': dsc})
results = model.evaluate(validate_dataset)
print('Test Loss ', results[0] )
print('Test Dice Coefficients ', results[1] )

# Load random validate image and segmentation
predict_image, predicted_seg = load_data_predict(validate_image_dir, 
                                                 validate_seg_dir, 
                                                 number_predict)
# Predict and visualise results
plt.figure(figsize=(12, 12))
figure_pos = 0
for i in range(number_predict):
    image = predict_image[i]
    seg = predicted_seg[i]
    predict = model.predict(image, batch_size=1)
    # Argmax to combines segmentation map to one mask
    predict = np.argmax(predict, axis=2)
    
    figure_pos += 1
    plt.subplot(number_predict, 3, figure_pos)
    plt.imshow(np.squeeze(image))
    plt.title('Original Image')

    figure_pos += 1
    plt.subplot(number_predict, 3, figure_pos)
    plt.imshow(np.squeeze(seg))
    plt.title('Original Mask')

    figure_pos += 1
    plt.subplot(number_predict, 3, figure_pos)
    plt.imshow(np.squeeze(predict))
    plt.title('Prediction')
plt.show()