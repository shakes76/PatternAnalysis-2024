import torch
from torchvision import transforms
from PIL import Image
from modules import GFNet
import os
import timm

# Set up device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('deit_base_patch16_224', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)  # Set the number of classes to 2
model.load_state_dict(torch.load('best_model.ckpt', map_location=device))  # Load the trained model weights
model = model.to(device)
model.eval()

# Define the preprocessing transformations
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(150),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess and predict a single image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transform(image)
    image = image.unsqueeze(0).to(device)
    return image

def predict_image(image_path):
    """
    Predicts the class of an image using a pre-trained model.

    Args:
        image_path (str): The file path to the image to be predicted.

    Returns:
        str: The predicted class of the image. 'AD' for Alzheimer's Disease, 'NC' for Normal Control.
    """
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_class = 'AD' if predicted.item() == 1 else 'NC'
    return predicted_class

# Directory with test images
test_dir = '/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test'

# Loop through test directory and predict for each image
results = []
for class_name in ['NC', 'AD']:
    class_dir = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        predicted_class = predict_image(image_path)
        results.append((image_name, predicted_class))
        print(f'Image: {image_name}, Predicted class: {predicted_class}')

# Save the results to a text file for documentation
with open("predictions.txt", "w") as f:
    for image_name, predicted_class in results:
        f.write(f"Image: {image_name}, Predicted class: {predicted_class}\n")

print("Predictions saved to predictions.txt")