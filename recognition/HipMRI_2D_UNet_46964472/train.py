import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import os
from dataset import *
from matplotlib import pyplot as plt

print(tf.keras.__version__)
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("Using CPU")
else:
    print("Using GPU")

img_path1 = os.path.join("keras_slices_data", "keras_slices_train", "case_004_week_0_slice_0.nii.gz")
img_path2 = os.path.join("keras_slices_data", "keras_slices_seg_train", "seg_004_week_0_slice_0.nii.gz")
image1 = load_data_2D([img_path1])
image2 = load_data_2D([img_path2])
plt.figure(1)
plt.imshow(image1[0])
plt.figure(2)
plt.imshow(image2[0])
plt.show()