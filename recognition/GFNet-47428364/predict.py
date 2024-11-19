from dataset import get_dataloaders
from train import evaluate
from modules import GFNet
import torch
import torch.nn as nn

# Select the GPU is available otherwise select the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Main method """
def main():
    # Load the saved model
    model = GFNet(embed_dim=256, patch_size=8)
    model.load_state_dict(torch.load("GFNet-Model.pth", weights_only=True))
    model.to(device)

    # Get the dataloaders from dataset.py
    _, test_dataloader = get_dataloaders()

    # Evaluate loaded model on the test data
    test_loss, test_accuracy = evaluate(model, test_dataloader, nn.CrossEntropyLoss())
    print(f"Test - Loss: {test_loss:.4f}, Accuracy: {100*test_accuracy:.2f}%")

if __name__ == "__main__":
    main()