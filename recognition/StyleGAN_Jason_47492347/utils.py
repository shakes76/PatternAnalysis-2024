import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# relative paths to dataset
test_path = '../ADNI_AD_NC_2D/AD_NC/test/NC'
train_path = '../ADNI_AD_NC_2D/AD_NC/train/NC'

# parameters
