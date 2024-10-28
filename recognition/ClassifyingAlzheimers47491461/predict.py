from modules import *
from train import *
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.fft


# Function to load the model
def load_model(state_dict_path=None):

    state_dict = torch.load(state_dict_path, map_location=device)

    model = GFNet(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=256,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.1
    ).to(device)

    print("Model loaded")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Function to set up the test DataLoader
def create_test_loader(batch_size=32):
    transform = transforms.Compose([
        AutoCropBlack(),  # Automatically crop black space
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_dataset = datasets.ImageFolder(root="C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\train", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to evaluate the model
def evaluate_model(model, test_loader):
    device = torch.device("cuda")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Main function to load the model, create the loader, and evaluate
def test_model_pipeline(state_dict_path, batch_size=32):
    device = torch.device("cuda")
    model = load_model(state_dict_path).to(device)
    test_loader = create_test_loader(batch_size)
    accuracy = evaluate_model(model, test_loader)
    return accuracy

# Usage example
# Poor generalisation and setup, training requires modification
state_dict_path = 'C:\\Users\\User\\Downloads\\best_gfnet.pt'
accuracy = test_model_pipeline(state_dict_path)
print(f"accuracy: {accuracy:.2f}%")
