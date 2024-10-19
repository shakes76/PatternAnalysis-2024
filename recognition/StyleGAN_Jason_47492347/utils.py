"""
Imports and utility functions are organised here.
"""


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from math import log2, sqrt
from settings import *


# set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
