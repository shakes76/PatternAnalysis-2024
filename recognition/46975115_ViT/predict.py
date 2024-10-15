import torch
from modules import VisionTransformer
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def predict(image_path, model_path='model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VisionTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    label_map = {0: 'Alzheimer\'s Disease (AD)', 1: 'Normal Cognitive (NC)'}
    predicted_label = label_map[prediction]
    
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.savefig('predicted_label.png')
    
    return predicted_label

if __name__ == "__main__":
    image_path = '/home/groups/comp3710/ADNI/AD_NC/test/AD/388747_97.jpeg'
    predicted_label = predict(image_path)
    print(f'Predicted Class: {predicted_label}')
