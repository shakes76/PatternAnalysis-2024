"""
Demonstrates the capabilities of the trainded GFNet image classification model.
The model predicts if an input image fits the AD/Alzheimer's or NC/normal classification.

Benjamin Thatcher 
s4784738    
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from modules import GFNet
from utils import get_parameters, get_prediction_image
from dataset import get_data_loader

def predict(image_path, model_path='best_model.pth', device='cuda'):
    """
    Loads an image, applies the GFNet model, and predicts whether it's 
    AD (ALzheimer's positive) or NC (Normal control).
    """
    # Define the image transformations to ensure the input image is compatible with the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    # Load the model architecture
    _, _, patch_size, embed_dim, depth, mlp_ratio, drop_rate, drop_path_rate, _, _ = get_parameters()
    model = GFNet(
        img_size=(224, 224),
        patch_size=patch_size,
        in_chans=1,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    ).to(device)

    # Load the saved model and set it up to evaluate
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Perform the prediction (1 for AD, 0 for normal)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return "Alzheimer's positive (AD)" if predicted.item() == 1 else "Normal (NC)"

if __name__ == "__main__":
    # Example image path
    #image_path = '../AD_NC/test/AD/388206_78.jpeg'
    image_path = '../AD_NC/test/NC/1185628_97.jpeg'
    image_path = get_prediction_image()

    # Set device (use CUDA if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run the prediction
    result = predict(image_path, model_path='best_model.pth', device=device)
    print(f'The predicted result is: {result}')
