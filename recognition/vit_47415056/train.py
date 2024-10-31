import torch
import torch.optim as optim
import torch.nn as nn
from modules import create_model
from dataset import get_dataloaders

def train_model(data_dir='/home/groups/comp3710/ADNI/AD_NC', batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes = get_dataloaders(data_dir, batch_size)