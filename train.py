import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time

from modules import VQVAE
import dataset


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='mri')
parser.add_argument('--train_dir', type=str, default='/HipMRI_study_keras_slices_data/keras_slices_train')
parser.add_argument('--test_dir', type=str, default='/HipMRI_study_keras_slices_data/keras_slices_test')

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

# training_data, validation_data, training_loader, validation_loader, x_train_var = dataset.load_data_and_data_loaders(
#     args.dataset, args.batch_size)
training_data, validation_data, training_loader, validation_loader = dataset.load_dataset(args)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

