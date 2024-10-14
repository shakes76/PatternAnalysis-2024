"""
Got inspiration from infer.py file of the following github repo:
https://github.com/shakes76/GFNet
"""

import torch
from modules import GFNet
from dataset import ADNIDataset


class AverageMeter(object):
    pass

class ProgressMeter(object):
    pass

def accuracy(output, target ,topk=(1,)):
    pass

def validate(val_loader, model, criterion):
    pass

if __name__ == '__main__':
    print("Main of Predict")