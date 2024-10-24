import tensorflow as tf
from tensorflow import keras
import os
from dataset import *
from modules import UNet2D
from utils import *
from keras.callbacks import ModelCheckpoint, History
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
learning_rate = 1e-3
epochs = 5
channels = 6


train_image_dir = os.path.join("keras_slices_data", "keras_slices_train")
train_seg_dir = os.path.join("keras_slices_data", "keras_slices_seg_train")
train_dataset = load_data_tf(train_image_dir, train_seg_dir, batch_size=batch_size)
test_image_dir = os.path.join("keras_slices_data", "keras_slices_test")
test_seg_dir = os.path.join("keras_slices_data", "keras_slices_seg_test")
test_dataset = load_data_tf(test_image_dir, test_seg_dir, batch_size=batch_size)

model = UNet2D((image_height, image_width, 1), 1024, channels=channels, activation="sigmoid")
# model.summary()
model.compile(optimizer='adam', loss=dsc_loss, metrics=[dsc])

history = model.fit(train_dataset, 
                    epochs=epochs, 
                    validation_data=test_dataset)

model.save('model.keras')

history_post_training = history.history

train_dice_coeff_list = history_post_training['dsc']
test_dice_coeff_list = history_post_training['val_dsc']

train_loss_list = history_post_training['loss']
test_loss_list = history_post_training['val_loss']

plt.figure(1)
plt.plot(test_loss_list, 'b-')
plt.plot(train_loss_list, 'r-')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss graph', fontsize=12)

plt.figure(2)
plt.plot(train_dice_coeff_list, 'b-')
plt.plot(test_dice_coeff_list, 'r-')

plt.xlabel('epochs')
plt.ylabel('dice similarity coefficient')
plt.title('DSC graph', fontsize=12)
plt.show()