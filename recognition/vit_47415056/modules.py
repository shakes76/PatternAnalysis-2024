import torch
import timm
import torch.nn as nn

def create_model(num_classes=2):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return model