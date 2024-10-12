import torch
from modules import GFNet
from dataset import ADNIDataset
from PIL import Image
import torchvision.transforms as transforms

# Got inspiration from infer.py file of github repo:
# https://github.com/shakes76/GFNet

def accuracy(output, target ,topk=(1,)):
    pass

def validate(val_loader, model, criterion):
    pass

if __name__ == '__main__':
    print("Main of Predict")