"""
The all in one place to modify hyperparameters and other key variables such as the datapath.

Made by Joshua Deadman
"""

DATAPATH = "./data"
MODELPATH = "./models"

# The percent of malignant data in each set (should add up to 1).
TRAINING = 0.7
TESTING = 0.2
VALIDATION = 0.1

# Hyperparameters
BATCH_SIZE = 32
WORKERS = 4
EPOCHS = 64
LEARNING_RATE = 0.0002
BETAS = (0.9,0.999)
LOSS_MARGIN = 1
BENIGN_LABEL = 0
MALIGNANT_LABEL = 1
