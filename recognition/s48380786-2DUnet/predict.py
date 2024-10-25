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

# Visualize a few predictions
num_examples = 3  # Number of examples to display
for i in range(num_examples):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Display Original Image
    axes[0].imshow(test_images[i][0], cmap='gray')
    axes[0].set_title("Original Image")

    # Convert the one-hot encoded ground truth mask to a single-channel mask
    single_channel_label = np.argmax(test_labels[i][0], axis=2)
    axes[1].imshow(single_channel_label, cmap='gray')
    axes[1].set_title("Ground Truth Mask")

    # Display the predicted mask
    axes[2].imshow(predicted_masks[i][0], cmap='gray')
    axes[2].set_title("Predicted Mask")

    plt.savefig(f"/home/Student/s4838078/model_predictions/predicted_mask_{i}.png")
    #plt.show()

