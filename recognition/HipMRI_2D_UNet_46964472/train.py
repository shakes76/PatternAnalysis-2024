"""
DataLoader, load and preprocess 2D slices of HipMRI Study on Prostate Cancer in Nifti file.

@author Thuan Pham - 46964472
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from dataset import load_data_tf
from modules import UNet2D
from utils import *
from matplotlib import pyplot as plt

# Check for access to GPU
print(tf.keras.__version__)
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("Using CPU")
else:
    print("Using GPU")

# hyper parameters
batch_size = 16
epochs = 5
if len(sys.argv) < 2:
    print("Error: not enough argument.")
    print("Usage: python train.py <dataset_dir> [epochs] [batch_size].")
    print("Default for epochs is 5 and batch_size is 16")
    quit()
else:
    dataset_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])
image_height = 256
image_width = 128
channels = 6

# Load train and test data
train_image_dir = os.path.join(dataset_dir, "keras_slices_train")
train_seg_dir = os.path.join(dataset_dir, "keras_slices_seg_train")
train_dataset = load_data_tf(train_image_dir, train_seg_dir, batch_size=batch_size)
test_image_dir = os.path.join(dataset_dir, "keras_slices_test")
test_seg_dir = os.path.join(dataset_dir, "keras_slices_seg_test")
test_dataset = load_data_tf(test_image_dir, test_seg_dir, batch_size=batch_size)

# Create 2D UNet Model
model = UNet2D((image_height, image_width, 1), 1024, channels=channels, activation="sigmoid")
model.compile(optimizer='adam', loss=dsc_loss, metrics=[dsc])
# Train
history = model.fit(train_dataset, 
                    epochs=epochs, 
                    validation_data=test_dataset)
# Save model
model.save('model.keras')

# Plot training result
history_post_training = history.history

train_dice_coeff_list = history_post_training['dsc']
test_dice_coeff_list = history_post_training['val_dsc']
train_loss_list = history_post_training['loss']
test_loss_list = history_post_training['val_loss']

plt.figure(1)
plt.plot(train_loss_list, 'b-', label='training')
plt.plot(test_loss_list, 'r-', label='testing')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss graph', fontsize=12)
plt.legend()

plt.figure(2)
plt.plot(train_dice_coeff_list, 'b-', label='training')
plt.plot(test_dice_coeff_list, 'r-', label='testing')
plt.xlabel('epochs')
plt.ylabel('dice similarity coefficient')
plt.title('DSC graph', fontsize=12)
plt.legend()

plt.show()