"""
Demonstrates the capabilities of the trainded GFNet image classification model

Benjamin Thatcher 
s4784738    
"""

import torch
from PIL import Image
from modules import get_model
import torchvision.transforms as transforms

def predict(image_path, model_path='gfnet_alzheimer_model.pth', device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.1155], [0.2224]),
    ])

    # Load the image and apply transforms
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    # Load the trained model
    model = get_model(num_classes = 2, pretrained = False)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return "AD" if pred.item() == 1 else "Normal"

if __name__ == "__main__":
    image_path = '/path/to/image.jpg'
    result = predict(image_path)
    print(f'The predicted result is: {result}')