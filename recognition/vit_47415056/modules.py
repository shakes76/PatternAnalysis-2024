import torch
import timm
import torch.nn as nn

def create_model(num_classes=2):
    """
    Creates and returns a Vision Transformer (ViT) model.

    Parameters:
    - num_classes (int): Number of output classes for the model. Default is 2.

    Returns:
    - model (torch.nn.Module): ViT model initialized with the specified number of output classes.
    """
    
    # Initialize a Vision Transformer (ViT) model from the timm library
    # The 'vit_tiny_patch16_224' model is used with 16x16 patches and a 224x224 input resolution
    # Set pretrained to False to initialize with random weights
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    
    return model