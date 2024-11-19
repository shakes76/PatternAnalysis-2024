import torch
import matplotlib.pyplot as plt
import numpy as np

from modules import UNet
from dataset import load_and_preprocess_data 


model_path = "/home/Student/s4838078/model_saves/unet_model.pth"

model = UNet()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

_, _, test_loader = load_and_preprocess_data()
# Get one batch of test images
test_images, test_labels = next(iter(test_loader))

# Move images to the same device as model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
test_images = test_images.to(device)

# Run prediction
with torch.no_grad():
    outputs = model(test_images)
    predicted_masks = (outputs > 0.5).float()  # Apply threshold to get binary mask

# Move data to CPU and convert to numpy for visualization
test_images = test_images.cpu().numpy()
predicted_masks = predicted_masks.cpu().numpy()
test_labels = test_labels.cpu().numpy()

# Number of channels
num_channels = test_labels.shape[-1]  # Assuming last dimension is channels
example_index = 0  # Index of the example image

for channel in range(num_channels):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Display Original Image
    print("Original Image shape:", test_images[example_index][0].shape)
    axes[0].imshow(test_images[example_index][0], cmap='gray')
    axes[0].set_title("Original Image")

    # Display the specific channel of the ground truth mask (squeezing extra dimension)
    ground_truth_mask = test_labels[example_index, 0, :, :, channel]  # Shape should be (256, 128)
    print(f"Ground Truth Mask - Channel {channel} shape:", ground_truth_mask.shape)
    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title(f"Ground Truth Mask - Channel {channel}")

    # Display the specific channel of the predicted mask (already in correct shape)
    predicted_mask = predicted_masks[example_index, channel, :, :]  # Shape should be (256, 128)
    print(f"Predicted Mask - Channel {channel} shape:", predicted_mask.shape)
    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title(f"Predicted Mask - Channel {channel}")

    plt.savefig(f"/home/Student/s4838078/model_predictions/predicted_mask_channel_{channel}.png")
    print("Predictions saved")
    plt.close(fig)






