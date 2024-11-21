import sys
sys.path.insert(1, './Modules')

import matplotlib.pyplot as plt
from modules import SiameseNetwork, LightweightSiamese
from dataset import FullMelanomaDataset, BalancedMelanomaDataset

import argparse

# hyperparameters
BATCH_SIZE = 256
IMAGE_SHAPE = (256, 256)
VALIDATION_SPLIT = 0.1
TESTING_SPLIT = 0.2
BALANCE_SPLIT = True


# pass command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--full-model', action='store_true')
parser.add_argument('--full-dataset', action='store_true')
args = parser.parse_args()

# get model
if args["full-model"]:
    model = SiameseNetwork(image_shape=IMAGE_SHAPE)
else:
    model = LightweightSiamese(image_shape=IMAGE_SHAPE)

# load full dataset
df_full = FullMelanomaDataset(    
    image_shape=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    testing_split=TESTING_SPLIT,
    balance_split=BALANCE_SPLIT
)

# load balanced dataset
df = BalancedMelanomaDataset(    
    image_shape=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    testing_split=TESTING_SPLIT,
    balance_split=BALANCE_SPLIT
)

# grab the relevent dataset
if args["full-dataset"]:
    dataset = df_full.dataset
    dataset_val = df_full.dataset_val
    dataset_test = df_full.dataset_test
else:
    dataset = df.dataset
    dataset_val = df.dataset_val
    dataset_test = df.dataset_test


# define siamese network
model = LightweightSiamese(image_shape=IMAGE_SHAPE)

# train the similarity
model.enable_wandb_similarity("melanoma-similarity")
model.train_similarity(dataset, dataset_val)

# plot TSNE
model.plot_embeddings(dataset_val, "tsne.png")

# train the classification
model.enable_wandb_classification("melanoma-classification")
model.enable_classification_checkpoints("./checkpoints", save_best_only=True)
model.train_classification(dataset, dataset_val)

# evaluate model
print("--- Testing Performance ---")
model.evaluate_classifier(dataset_test)

# plot training curves
model.plot_training_classification("classification.png")
model.plot_training_similarity("similarity.png")

# plot graphs
plt.show()