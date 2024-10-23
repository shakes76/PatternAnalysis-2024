import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time

from modules import improved_3d_unet_model
from dataset import DOWNSIZE_FACTOR, load_mri_data

#Path for MRI data
DATA_PATH = "C:/Users/nk200/Downloads/HipMRI_study_complete_release_v1/" #This is my path, please change when using

BATCH_LENGTH = 2
BUFFER_SIZE = 64

EPOCHS = 10

def train_model():
    """
    Train the model and calculate training, validation and test results.
    """
    
    #Load and batch data
    train_dataset, test_dataset, validate_dataset = load_mri_data(DATA_PATH, False)
    train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_LENGTH).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = validate_dataset.batch(BATCH_LENGTH)
    test_batches = test_dataset.batch(BATCH_LENGTH)

if __name__ == "__main__":
    
    train_model()