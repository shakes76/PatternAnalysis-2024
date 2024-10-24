import torch
from modules import UNet
from dataset import ProstateCancerDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load the model
model = UNet().cuda()
model.load_state_dict(torch.load('unet_model.pth'))
model.eval()

# Load your dataset (replace with actual paths)
test_images = [...]
test_dataset = ProstateCancerDataset(test_images, test_images)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def predict():
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.cuda()
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
