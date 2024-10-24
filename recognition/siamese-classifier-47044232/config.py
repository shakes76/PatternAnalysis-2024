"""
The all in one place to modify hyperparameters and other key variables such as the datapath.

Made by Joshua Deadman
"""
# Datapath should be of structure:
# ./data
#       train-metadata.csv
#       /train-image
#           /images
#               all images
DATAPATH = "./data"
MODELPATH = "./models"
IMAGEPATH = "./images"

# The percent of malignant data in each set (should add up to 1).
TRAINING = 0.7
TESTING = 0.2
VALIDATION = 0.1

# Hyperparameters
BATCH_SIZE = 32
WORKERS = 4
EPOCHS_SIAMESE = 140
EPOCHS_CLASSIFIER = 50
LR_SIAMESE = 0.0002
LR_CLASSIFIER = 0.002
BETAS = (0.9,0.999)
LOSS_MARGIN = 1
