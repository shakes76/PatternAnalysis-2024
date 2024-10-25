import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing function with visualization


def test_and_visualize(model, test_loader):
    model.eval()
    with torch.no_grad():
        for test_images, test_masks in test_loader:
            test_images, test_masks = test_images.to(
                device), test_masks.to(device)
            test_outputs = model(test_images)
            predicted_masks = torch.argmax(test_outputs, dim=1)
            middle_slice_idx = test_images.shape[2] // 2
            # Visualize the first example in the batch
            plt.figure(figsize=(12, 6))

            # Input image
            plt.subplot(1, 3, 1)
            plt.imshow(test_images[0, 0, middle_slice_idx].cpu(), cmap='gray')
            plt.title("Input Image")

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(test_masks[0, 0, middle_slice_idx].cpu(), cmap='gray')
            plt.title("Ground Truth Mask")

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[0, middle_slice_idx].cpu(), cmap='gray')
            plt.title("Predicted Mask")

            plt.tight_layout()
            plt.savefig("output.png")
            plt.show()
            break
