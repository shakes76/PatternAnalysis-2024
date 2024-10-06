import os
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_data_loaders
from modules import get_model, get_triplet_loss
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime

# Make sure GPU is available, added print statement to check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
csv_file = 'ISIC_2020_Training_GroundTruth_v2.csv'
img_dir = 'data/ISIC_2020_Training_JPEG/train/'

# Premilimary hyperparameters
batch_size = 32
embedding_dim = 128
learning_rate = 1e-3
num_epochs = 50

# Data loading
train_loader, val_loader = get_data_loaders(csv_file=csv_file, img_dir=img_dir, batch_size=batch_size)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_directory = f"runs/{current_time}"
os.makedirs(run_directory, exist_ok=True)

def save_plot(plt, file):
    plt.savefig(os.path.join(run_directory, file))
    plt.close()

