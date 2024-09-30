import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from torch.utils.data import Dataset
import time
import numpy as np
import nibabel as nib
from tqdm import tqdm
import utils
import glob

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

files = sorted(glob.glob(f"{file_path}", recursive = True))
print(files)