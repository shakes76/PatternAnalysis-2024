import torch
from torchvision import transforms
from PIL import Image
from modules import GFNet
import os

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GFNet(img_size=224, patch_size=16, num_classes=2, embed_dim=768, depth=12).to(device)
model.load_state_dict(torch.load('gfnet_model.ckpt'))
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = data_transform(image)
    image = image.unsqueeze(0).to(device)
    return image

# Make prediction
def predict_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_class = 'AD' if predicted.item() == 1 else 'CN'
    return predicted_class

# Directory containing test images
test_dir = '/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test'

for class_name in ['AD', 'NC']:
    class_dir = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        predicted_class = predict_image(image_path)
        print(f'Image: {image_name}, Predicted class: {predicted_class}')