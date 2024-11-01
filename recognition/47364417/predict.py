import os
import torch
from modules import create_model
from dataset import get_dataloaders

def predict():
    """
    Loads a saved model (or trains a new one if none exists) and tests it on the testing data.
    """
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define directories and model path.
    data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    checkpoints_dir = 'checkpoints'
    final_model_path = os.path.join(checkpoints_dir, 'final_model.pth')

    # Get dataloaders and class names.
    dataloaders, class_names = get_dataloaders(data_dir)
    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Initialize the model.
    model = create_model(num_classes)
    model = model.to(device)

    # Load the saved model if possible.
    if os.path.exists(final_model_path):
        print(f'Loading saved model from {final_model_path}')
        model.load_state_dict(torch.load(final_model_path, map_location=device))
    else:
        print('No saved model found.')