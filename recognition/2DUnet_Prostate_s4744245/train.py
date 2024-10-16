import numpy as np
from matplotlib import pyplot as plt
import glob
import tensorflow as tf

from dataset import load_data_2D
from modules import unet_model


# Get all file paths for train, test, and validate sets
#train_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_train/*.nii.gz')
#test_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_test/*.nii.gz')
#validate_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_validate/*.nii.gz')

#seg_train_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_train/*.nii.gz')
#seg_test_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_test/*.nii.gz')
#seg_validate_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_validate/*.nii.gz')

train_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train/*.nii.gz')
test_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test/*.nii.gz')
validate_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate/*.nii.gz')

seg_train_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/*.nii.gz')
seg_test_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test/*.nii.gz')
seg_validate_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/*.nii.gz')


# Load the images using the load_data_2D function
images_train = load_data_2D(train_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_test = load_data_2D(test_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_validate = load_data_2D(validate_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))

images_seg_train = load_data_2D(seg_train_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_seg_test = load_data_2D(seg_test_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_seg_validate = load_data_2D(seg_validate_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))

# print the shapes of the loaded datasets
print(f"Training data shape: {images_train.shape}")
print(f"Test data shape: {images_test.shape}")
print(f"Validation data shape: {images_validate.shape}")
print(f"Segement Training data shape: {images_seg_train.shape}")
print(f"Segement Test data shape: {images_seg_test.shape}")
print(f"Segement Validation data shape: {images_seg_validate.shape}")

#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Initialize the U-Net model with the adjusted input size
model = unet_model(input_size=(256, 128, 1))

#temp compile to check model summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to verify the architecture
model.summary()