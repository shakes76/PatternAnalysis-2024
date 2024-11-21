# predict.py
# Script to make predictions using the trained Image Classifier.
# author: Harrison Martin

import torch
from torchvision import transforms
from PIL import Image
import argparse
from modules import ImageClassifier

def main():
    parser = argparse.ArgumentParser(description='Predict skin lesion classification.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--model_path', type=str, default='results/best_classifier_model.pth', help='Path to the trained model.')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Define transforms (same as validation transforms)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        if prob >= 0.5:
            pred_class = 'Malignant'
        else:
            pred_class = 'Benign'

    print(f"Predicted Class: {pred_class}")
    print(f"Probability of being Malignant: {prob:.4f}")

if __name__ == '__main__':
    main()
