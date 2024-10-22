# predict.py

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules import VisionTransformer
import os


def load_model(model_path, device, img_size=224, patch_size=16, emb_size=768, num_heads=12, depth=12, ff_dim=3072, num_classes=2, dropout=0.1, cls_token=True):
    """
    Loads the trained Vision Transformer model.

    Args:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model on.
        Other args: Model architecture parameters.

    Returns:
        nn.Module: Loaded model.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        emb_size=emb_size,
        num_heads=num_heads,
        depth=depth,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout,
        cls_token=cls_token
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, img_size=224):
    """
    Preprocesses the input image.

    Args:
        image_path (str): Path to the image.
        img_size (int): Size to resize the image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image