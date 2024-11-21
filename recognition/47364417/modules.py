import torch.nn as nn
import timm

def create_model(num_classes):
    """
    Creates and returns a Vision Transformer model with the specified number of classes.
    """
    # Initialize the Vision Transformer model without pretrained weights.
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    return model
