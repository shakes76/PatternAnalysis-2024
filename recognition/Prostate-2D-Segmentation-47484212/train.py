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

TEST_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
TEST_SEG_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

# create list of paths to each of the datasets
trainPaths = get_all_paths(TRAIN_PATH)
trainSegPaths = get_all_paths(TRAIN_SEG_PATH)

validationPaths = get_all_paths(VALIDATION_PATH)
validationSegPaths = get_all_paths(VALIDATION_SEG_PATH)

testPaths = get_all_paths(TEST_PATH)
testSegPaths = get_all_paths(TEST_SEG_PATH)
# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# load or initialize model
unet = UNetSegmentation(MODEL_PATH)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.Dice()

# testing
preTrain_loss = tf.keras.metrics.Mean()

loss_fn = tf.keras.losses.Dice()

# batching test to reduce memory usage
x_test_batches, y_test_batches = batch_paths(testPaths, testSegPaths, 32)
for test_x_paths, test_y_paths in zip(x_test_batches, y_test_batches):
    x_test = load_data_2D(test_x_paths, normImage=True)
    y_test = load_data_2D(test_y_paths, categorical=True)

    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    test_logits = unet.model(x_test_tensor, training=False)
    test_loss_value = loss_fn(y_test_tensor, test_logits)
    preTrain_loss.update_state(test_loss_value)


# track previous epoch loss to find convergence
prev_loss = 0

epochLoss = [] # keep track of validation loss at each epoch

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
        y_val = load_data_2D(val_y_paths, categorical=True)
        
        x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
        y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

        val_logits = unet.model(x_val_tensor, training=False)
        val_loss_value = loss_fn(y_val_tensor, val_logits)
        val_loss.update_state(val_loss_value)
    
    print(f"Validation loss after epoch {epoch+1}: {val_loss.result().numpy():.4f}")
    unet.model.save(MODEL_PATH)
    epochLoss.append(float(val_loss.result()))

print(f"completed {EPOCHS} epochs, final validation loss was {loss_value}")

# testing
test_loss = tf.keras.metrics.Mean()

loss_fn = tf.keras.losses.Dice()

prostateLoss = [] # loss specifically for the prostate label

# batching test to reduce memory usage
x_test_batches, y_test_batches = batch_paths(testPaths, testSegPaths, 32)
for test_x_paths, test_y_paths in zip(x_test_batches, y_test_batches):
    x_test = load_data_2D(test_x_paths, normImage=True)
    y_test = load_data_2D(test_y_paths, categorical=True)

    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    test_logits = unet.model(x_test_tensor, training=False)
    # calculate loss for prostate class
    prosPredMasks = test_logits[:, :, :, 0]
    prosRealMasks = y_test_tensor[:, :, :, 0]
    prosDice = 1 -  (2 * tf.reduce_sum(prosPredMasks * prosRealMasks) + 1e-6) / (tf.reduce_sum(prosPredMasks + prosRealMasks) + 1e-6)
    prostateLoss.append(prosDice)
    
    test_loss_value = loss_fn(y_test_tensor, test_logits)
    test_loss.update_state(test_loss_value)

print(f"prostateLoss: {np.mean(prostateLoss)}")
print(f"untrained loss: {preTrain_loss.result().numpy():.4f}")
print(f"validation loss at each epoch: {epochLoss}")
print(f"test loss: {test_loss.result().numpy():.4f}")
