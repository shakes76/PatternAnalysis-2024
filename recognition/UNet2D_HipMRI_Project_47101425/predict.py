import torch
import matplotlib.pyplot as plt
from modules import UNet  # Assuming you have a UNet architecture in 'modules'
from dataset import MedicalImageDataset
from torch.utils.data import DataLoader

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('./model.pth'))  # Replace with your model's path
model.eval()  # Set the model to evaluation mode

# Define test loader
test_image_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/keras_slices_seg_test"  # Replace with the directory containing your test images
test_dataset = MedicalImageDataset(image_dir=test_image_dir, normImage=True, load_type='2D')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)


# Dice coefficient for segmentation accuracy
def dice_coefficient(pred, target, epsilon=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return dice.item()


# Function to test the model and print accuracy metrics
def test_model(model, test_loader):
    print("> Testing")
    model.eval()  # Set the model to evaluation mode
    dice_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Pass the images through the model
            outputs = model(images)
            predicted_masks = (outputs > 0.5).float()  # Apply threshold for binary segmentation

            # Calculate dice coefficient for each image in the batch
            for pred, target in zip(predicted_masks, labels):
                dice = dice_coefficient(pred, target)
                dice_scores.append(dice)

    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f'Average Dice Coefficient: {avg_dice:.4f}')

    return dice_scores


# Visualization code
def visualize_predictions(model, test_loader, num_images=4):
    model.eval()

    # Fetch a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        predicted_masks = (outputs > 0.5).float()  # Apply threshold for binary predictions

    # Visualize the original images, ground truth, and predictions
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
    for i in range(min(num_images, len(images))):
        ax_image, ax_gt, ax_pred = axes[i]

        # Display the original image
        ax_image.imshow(images[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
        ax_image.set_title(f'Image {i+1}')
        ax_image.axis('off')

        # Display the ground truth mask
        ax_gt.imshow(labels[i].cpu().numpy(), cmap='gray')
        ax_gt.set_title(f'Ground Truth {i+1}')
        ax_gt.axis('off')

        # Display the predicted mask
        ax_pred.imshow(predicted_masks[i].cpu().numpy(), cmap='gray')
        ax_pred.set_title(f'Prediction {i+1}')
        ax_pred.axis('off')

    plt.tight_layout()
    plt.show()


# Test the model and visualize the results
test_model(model, test_loader)
visualize_predictions(model, test_loader, num_images=4)
