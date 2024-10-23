"""
This file contains various constants and fixed parameters used in the training, validation and 
testing of the model.

Abdullah Badat (47022173), abdullahbadat27@gmail.com
"""

# Data split parameters
VALID_START = 195
TEST_START = 203
DEBUG = 6
LOAD_SIZE = 50

# Data loading parameters
NUM_WORKERS = 2

# Fixed model parameters
IN_CHANNELS = 1 # greyscale
BASE_N_FILTERS = 8
N_CLASSES = 6

# Training default parameters
LR_D = 1e-3
WD_D = 1e-2
SS_D = 10
G_D = 0.1

# Output dimensions
WIDTH = 128
HEIGHT = 128
DEPTH = 64