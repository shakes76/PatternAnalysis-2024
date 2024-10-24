import torch
from modules import UNet
from dataset import ProstateCancerDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = UNet().to(device)
model.load_state_dict(torch.load('unet_model.pth', map_location=device))  # Ensure model is loaded to the right device
model.eval()

# Load your test dataset (replace with actual paths)
test_image_dir = r'C:/Users/11vac/Desktop/3710 Report/HipMRI_study_keras_slices_data/keras_slices_test'
test_mask_dir = r'C:/Users/11vac/Desktop/3710 Report/HipMRI_study_keras_slices_data/keras_slices_seg_test'

test_dataset = ProstateCancerDataset(test_image_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def predict():
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(0).cpu().numpy()

            # Visualize prediction
            plt.subplot(1, 2, 1)
            plt.imshow(images.squeeze(0).cpu().numpy(), cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(outputs, cmap='gray')
            plt.show()

if __name__ == "__main__":
    predict()
