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
number_predict = 5

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
true_image, predict_image, predicted_seg = load_data_predict(validate_image_dir, 
                                                 validate_seg_dir, 
                                                 number_predict)
# Predict
predict = model.predict(predict_image)
# Argmax to reverse one hot encoding
predict = np.argmax(predict, axis=3)

# Visualise results
plt.figure()
figure_pos = 0
for i in range(number_predict):    
    figure_pos += 1
    plt.subplot(3, number_predict, i + 1)
    plt.imshow(true_image[i])
    plt.title('Original Image')

    figure_pos += 1
    plt.subplot(3, number_predict, number_predict + i + 1)
    plt.imshow(predicted_seg[i])
    plt.title('Original Mask')

    figure_pos += 1
    plt.subplot(3, number_predict, number_predict * 2 + i + 1)
    plt.imshow(predict[i])
    plt.title('Prediction')
plt.show()