import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from modules import UNet  # Ensure your U-Net model is defined in modules.py
from dataset import create_dataloader  # Import the DataLoader creation function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Loads the trained U-Net model from the specified path."""
    model = UNet()  # Initialize the model
    model.load_state_dict(torch.load(model_path))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(model, image):
    """Runs the model on a single image and returns the predicted mask."""
    with torch.no_grad():  # Disable gradient calculation
        image_tensor = image.unsqueeze(0)  # Add batch dimension
        prediction = model(image_tensor)  # Get model prediction
        predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()  # Get the predicted class labels
    return predicted_mask

def save_prediction(predicted_mask, output_path):
    """Saves the predicted mask as a PNG image."""
    plt.imshow(predicted_mask, cmap='gray')  # Display the predicted mask
    plt.axis('off')  # Turn off axis labels
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()  # Close the plot to free up memory

def dice_score(predicted, target, num_classes):
    """Calculates the Dice score for each class."""
    dice_scores = []  # Initialize a list to store Dice scores for each class
    for i in range(num_classes):  # Loop over each class
        intersection = np.sum((predicted == i) & (target == i))
        union = np.sum(predicted == i) + np.sum(target == i)
        if union == 0:
            dice = 1.0  # Perfect match when both predicted and target have no pixels for the class
        else:
            dice = 2.0 * intersection / union  # Compute Dice score
        dice_scores.append(dice)  # Append the Dice score to the list
    return dice_scores

def process_and_evaluate(model, dataloader, output_dir, num_classes):
    """Processes the dataloader for predictions, saves results, and computes the Dice score."""
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Iterate over the dataloader
    for idx, (image_tensor, label_tensor) in enumerate(dataloader):
        # Move tensors to the appropriate device (CPU or GPU)
        image_tensor = image_tensor.to(device)  # Ensure image tensor is on the same device as model
        label_tensor = label_tensor.numpy()  # Convert label tensor to numpy for evaluation
        
        predicted_mask = predict_image(model, image_tensor)  # Get the predicted segmentation mask
        
        # Save the predicted segmentation mask
        output_path = os.path.join(output_dir, f'predicted_mask_{idx}.png')  # Save output as PNG
        save_prediction(predicted_mask, output_path)

        # Calculate Dice score for each class
        dice_scores = dice_score(predicted_mask, label_tensor[0], num_classes)  # Assuming single-channel labels
        print(f"Dice scores for image {idx}: {dice_scores}")

def main():
    """Main function to run inference and evaluate the model on test data."""
    model_path = r""  # Path to the trained model                                  PLEASE ADD  #################
    test_images_dir = r""  # Directory containing test MRI images                  PLEASE ADD  #################
    test_labels_dir = r""  # Directory containing ground truth labels              PLEASE ADD  #################
    output_dir = r""  # Directory to save predicted masks                          PLEASE ADD  #################
    
    # Create DataLoader for test images and labels
    test_dataloader = create_dataloader(test_images_dir, test_labels_dir, batch_size=1, max_images=1000000, normImage=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model = load_model(model_path)  # Load the trained model
    model.to(device)  # Move model to the appropriate device
    
    num_classes = 6  # Set the number of classes for segmentation
    
    # Process the images and evaluate the model
    process_and_evaluate(model, test_dataloader, output_dir, num_classes)

if __name__ == '__main__':
    main()  # Run the main function
