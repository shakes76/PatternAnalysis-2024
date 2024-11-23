from module import GFNet
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import logging
import math
from functools import partial
from collections import OrderedDict
import torch.optim as optim


# Initialize logger
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_logger.info(f"Using device: {device}")

# Data Preparation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load ADNI dataset
train_dataset = datasets.ImageFolder(
    root=r"ADNI_AD_NC_2D\AD_NC\train", transform=transform
)
test_dataset = datasets.ImageFolder(
    root=r"ADNI_AD_NC_2D\AD_NC\test", transform=transform
)

# Split the test dataset into validation and test subsets
val_size = int(0.8 * len(test_dataset))
test_size = len(test_dataset) - val_size
val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Output the sizes of the datasets
_logger.info(f"Total training images: {len(train_dataset)}")
_logger.info(f"Total validation images: {len(val_dataset)}")
_logger.info(f"Total test images: {len(test_dataset)}")

# Model Initialization
model = GFNet(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=2,
    embed_dim=768,
    depth=12,
    mlp_ratio=4,
    drop_rate=0,
    drop_path_rate=0.0,
)
model.head = nn.Linear(model.num_features, 2)  # Assuming binary classification
model.to(device)  # Move model to GPU
