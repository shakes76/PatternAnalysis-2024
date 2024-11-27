import torch
from timm import create_model

# Configure the device to use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model():
    """
    Initializes and returns a Vision Transformer (ViT) model.

    The model uses a smaller architecture variant, 'vit_small_patch16_224',
    without pretrained weights and is configured for 2 output classes.

    Returns:
    - model (torch.nn.Module): ViT model moved to the specified device.
    """
    
    # Create a Vision Transformer (ViT) model using 'vit_small_patch16_224' 
    model = create_model("vit_small_patch16_224", pretrained=False, num_classes=2)
    
    return model.to(DEVICE)