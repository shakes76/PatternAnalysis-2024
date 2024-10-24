import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time

from dataset import load_mri_data
from train import DATA_PATH, BATCH_LENGTH, BUFFER_SIZE

def predict_model():
    """
    Make predictions using the trained model.
    """
    
    #Load and batch test data
    test_dataset = load_mri_data(DATA_PATH, True)[1]
    test_batches = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_LENGTH)

if __name__ == "__main__":
    
    predict_model()