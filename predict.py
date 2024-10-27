from train import train_model, evaluate_model
from dataset import device, train_loader, val_loader, test_loader
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
model.to(device)

trained_model = train_model(
    model, train_loader, val_loader, num_epochs=25, learning_rate=0.001
)

test_accuracy = evaluate_model(trained_model, test_loader)

torch.save(trained_model.state_dict(), "gfnet_adni_model2.pth")

# Load the model for inference
model.load_state_dict(torch.load("gfnet_adni_model.pth"))
model.to(device)  # Ensure the model is on the GPU for inference
model.eval()
