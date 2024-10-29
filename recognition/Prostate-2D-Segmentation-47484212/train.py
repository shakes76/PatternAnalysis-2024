from modules import UNetSegmentation
import tensorflow as tf
import os
from dataset import load_data_2D, get_all_paths, batch_paths
import random
import numpy as np

MODEL_PATH = "unetSeg.keras"

TRAIN_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
TRAIN_SEG_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"

VALIDATION_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
VALIDATION_SEG_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

TEST_PATH = "C:/Users/rjmah/Documents/Sem2 2024/COMP3710/HipMRI_study_keras_slices_data/keras_slices_test"
TEST_SEG_PATH = "C:/Users/rjmah/Documents/Sem2 2024/COMP3710/HipMRI_study_keras_slices_data/keras_slices_seg_test"

# create list of paths to each of the datasets
trainPaths = get_all_paths(TRAIN_PATH)
trainSegPaths = get_all_paths(TRAIN_SEG_PATH)

validationPaths = get_all_paths(VALIDATION_PATH)
validationSegPaths = get_all_paths(VALIDATION_SEG_PATH)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# load or initialize model
unet = UNetSegmentation(MODEL_PATH)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.Dice()

# testing save
unet.model.save(MODEL_PATH)

for epoch in range(EPOCHS):
    # randomly sample without replacement to divide the dataset into batches of BATCH_SIZE for each epoch

    x_batch_paths, y_batch_paths = batch_paths(trainPaths, trainSegPaths, BATCH_SIZE)

    ### NOTE: following code is taken from tensorflow documentation
    # Iterate over the batches of the dataset
    for step in range(len(x_batch_paths)):
        # current batch
        x_batch = load_data_2D(x_batch_paths[step], normImage=True)
        y_batch = load_data_2D(y_batch_paths[step], categorical=True)

        x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        # use gradient tape for auto differentiation
        with tf.GradientTape() as tape:
            pred = unet.call(x_tensor, training=True) 
            loss_value = loss_fn(y_tensor, pred)
        # get the gradients and apply to model
        grads = tape.gradient(loss_value, unet.get_trainable_weights())
        optimizer.apply_gradients(zip(grads, unet.get_trainable_weights()))

    # Validation at the end of each epoch (batched to reduce memory)
    val_loss = tf.keras.metrics.Mean()

    # batching validation to reduce memory usage
    x_validation_batches, y_validation_batches = batch_paths(validationPaths, validationSegPaths, BATCH_SIZE)
    for val_x_paths, val_y_paths in zip(x_validation_batches, y_validation_batches):
        x_val = load_data_2D(val_x_paths, normImage=True)
        y_val = load_data_2D(val_y_paths)
        
        x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
        y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

        val_logits = unet.model(x_val_tensor, training=False)
        val_loss_value = loss_fn(y_val_tensor, val_logits)
        val_loss.update_state(val_loss_value)
    
    print(f"Validation loss after epoch {epoch+1}: {val_loss.result().numpy():.4f}")
    unet.model.save(MODEL_PATH)

print(f"completed {EPOCHS} epochs, final loss was {loss_value}")
