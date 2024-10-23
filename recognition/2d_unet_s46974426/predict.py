import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import UNet
from dataset import load_data_2D  # Ensure to adjust this based on your structure

def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        return torch.sigmoid(output).squeeze(0)

if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load('model_checkpoint.pth'))  # Load your trained model

    # Load a sample image for prediction
    sample_image_paths = ["path/to/sample/image.nii.gz"]
    sample_images = load_data_2D(sample_image_paths)

    preds = predict(model, sample_images[0])  # Assuming sample_images is the first loaded sample

    # Visualization
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(sample_images[0][0, :, :], cmap='gray')  # Assuming single channel

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(preds.numpy(), cmap='gray')
    
    plt.show()
