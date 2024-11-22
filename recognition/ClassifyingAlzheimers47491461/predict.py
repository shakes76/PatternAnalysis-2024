from modules import *
from train import *
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.fft
import os

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to train and save the model
def train_and_save_model(state_dict_path):
    # Initialize the model with default hyperparameters
    model = GFNet(
        img_size=224, patch_size=14, in_chans=1, num_classes=2, embed_dim=256, depth=12,
        mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1
    ).to(device)

    # Train the model (Assuming train_model returns the trained model)
    trained_model = train_model(model)

    # Save the trained model's state dict to the current directory
    torch.save(trained_model.state_dict(), state_dict_path)
    print(f"Model trained and saved at {state_dict_path}")

    return trained_model

# Function to load the model
def load_model(state_dict_path):
    # Initialize the model architecture
    model = GFNet(
        img_size=224, patch_size=14, in_chans=1, num_classes=2, embed_dim=256, depth=12,
        mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1
    ).to(device)

    # Load the saved state dict
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {state_dict_path}")

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
    test_dataset = datasets.ImageFolder(
        root="C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\test",
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to evaluate the model
def evaluate_model(model, test_loader):
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

# Main function to train, save, load, and evaluate the model
def test_model_pipeline(state_dict_path, batch_size=32):
    # Train and save the model
    train_and_save_model(state_dict_path)

    # Load the model
    model = load_model(state_dict_path)

    # Create the test DataLoader
    test_loader = create_test_loader(batch_size)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)
    return accuracy

if __name__ == "__main__":
    # Define the path to save and load the model's state dict

    _, _, _ = train_model()
    state_dict_path = os.path.join(os.getcwd(), 'model_state.pth')

    # Run the test model
    accuracy = test_model_pipeline(state_dict_path)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
