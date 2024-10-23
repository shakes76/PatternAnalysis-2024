import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

print(tf.keras.__version__)
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("Using CPU")
else:
    print("Using GPU")

