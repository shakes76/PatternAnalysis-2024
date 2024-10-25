import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from modules import UNet, dice_coefficient  # Importing the UNet model and Dice coefficient
from dataset import load_and_resize_images, prepare_dataloaders  # Assuming you created dataset.py

