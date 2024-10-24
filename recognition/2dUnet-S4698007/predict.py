import torch
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from modules import UNet  # Import the U-Net model from the modules.py file
from dataset import preprocess_image  # Import the preprocessing function for the MRI images

def load_model(model_path):
    """Loads the trained U-Net model from the specified path.

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        model: The loaded U-Net model ready for inference.
    """
    model = UNet(n_classes=6)  # Instantiate the U-Net model with the number of classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load model weights
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(model, image):
    """Runs the model on a single image and returns the predicted mask.

    Args:
        model: The trained U-Net model.
        image (numpy.ndarray): The input MRI image for prediction.

    Returns:
        numpy.ndarray: The predicted mask indicating segmented regions.
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        # Convert image to a tensor and add batch and channel dimensions
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        prediction = model(image_tensor)  # Get model prediction
        predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()  # Get the predicted class labels
    return predicted_mask

def load_mri_image(image_path):
    """Loads and preprocesses the MRI image from a NIfTI file.

    Args:
        image_path (str): Path to the NIfTI file.

    Returns:
        numpy.ndarray: The processed MRI slice ready for prediction.
    """
    img = nib.load(image_path)  # Load the NIfTI image
    img_data = img.get_fdata()  # Get the data from the NIfTI file
    img_slice = img_data[:, :, img_data.shape[2] // 2]  # Select the central slice
    processed_img = preprocess_image(img_slice)  # Preprocess the image
    return processed_img

def save_prediction(predicted_mask, output_path):
    """Saves the predicted mask as a PNG image.

    Args:
        predicted_mask (numpy.ndarray): The predicted segmentation mask.
        output_path (str): Path where the predicted mask will be saved.
    """
    plt.imshow(predicted_mask, cmap='gray')  # Display the predicted mask
    plt.axis('off')  # Turn off axis labels
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()  # Close the plot to free up memory

def dice_score(predicted, target, num_classes):
    """Calculates the Dice score for each class.

    Args:
        predicted (numpy.ndarray): The predicted segmentation mask.
        target (numpy.ndarray): The ground truth segmentation mask.
        num_classes (int): The number of classes in the segmentation task.

    Returns:
        list: A list of Dice scores for each class.
    """
    dice_scores = []  # Initialize a list to store Dice scores for each class
    for i in range(num_classes):  # Loop over each class
        # Calculate intersection and union for the Dice score
        intersection = np.sum((predicted == i) & (target == i))
        union = np.sum(predicted == i) + np.sum(target == i)
        if union == 0:
            dice = 1.0  # Perfect match when both predicted and target have no pixels for the class
        else:
            dice = 2.0 * intersection / union  # Compute Dice score
        dice_scores.append(dice)  # Append the Dice score to the list
    return dice_scores

def process_and_evaluate(model, image_path, label_path, output_path, num_classes):
    """Loads the image and label, predicts the mask, saves the result, and computes the Dice score.

    Args:
        model: The trained U-Net model.
        image_path (str): Path to the test image.
        label_path (str): Path to the ground truth label image.
        output_path (str): Path to save the predicted mask.
        num_classes (int): The number of classes for segmentation.

    Returns:
        list: Dice scores for each class.
    """
    image = load_mri_image(image_path)  # Load and preprocess the MRI image
    label = load_mri_image(label_path)  # Load and preprocess the ground truth label
    
    predicted_mask = predict_image(model, image)  # Get the predicted segmentation mask
    
    # Save the predicted segmentation mask
    save_prediction(predicted_mask, output_path)

    # Calculate Dice score for each class
    dice_scores = dice_score(predicted_mask, label, num_classes)
    
    return dice_scores  # Return the Dice scores

def main(args):
    """Main function to run inference and evaluate the model on test images.

    Args:
        args: Parsed command-line arguments.
    """
    model = load_model(args.model)  # Load the trained model
    
    num_classes = 6  # Set the number of classes for segmentation
    
    # Get lists of test images and labels, sorted to ensure correspondence
    test_images = sorted(os.listdir(args.test_images_dir))
    test_labels = sorted(os.listdir(args.test_labels_dir))
    
    # Check if the number of images and labels match
    if len(test_images) != len(test_labels):
        print("Error: Number of test images and test labels must be the same.")
        return

    total_dice_scores = np.zeros((len(test_images), num_classes))  # Initialize an array to store Dice scores for all images

    # Process each test image and calculate Dice scores
    for i, (img_name, label_name) in enumerate(zip(test_images, test_labels)):
        img_path = os.path.join(args.test_images_dir, img_name)  # Construct the path for the test image
        label_path = os.path.join(args.test_labels_dir, label_name)  # Construct the path for the corresponding label
        output_path = os.path.join(args.output_dir, f"pred_{img_name}.png")  # Set the output path for saving predictions
        
        # Process the image, make predictions, and evaluate the results
        dice_scores = process_and_evaluate(model, img_path, label_path, output_path, num_classes)
        total_dice_scores[i] = dice_scores  # Store the Dice scores for this image
        
        # Print Dice scores for the current image
        print(f"Dice scores for {img_name}: {dice_scores}")

    # Compute average Dice scores across all test images
    avg_dice_scores = np.mean(total_dice_scores, axis=0)
    print(f"Average Dice scores across all test images: {avg_dice_scores}")

    # Print detailed average Dice scores for each class
    for class_idx in range(num_classes):
        class_avg_score = np.mean(total_dice_scores[:, class_idx])  # Calculate average Dice score for the current class
        print(f"Average Dice score for class {class_idx}: {class_avg_score}")

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run inference on MRI images using a trained U-Net model.")
    parser.add_argument('--test_images_dir', type=str, required=True, help="Directory containing test MRI NIfTI files.")
    parser.add_argument('--test_labels_dir', type=str, required=True, help="Directory containing ground truth labels for the test images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output segmentation masks.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained U-Net model.")
    
    args = parser.parse_args()  # Parse the command-line arguments
    main(args)  # Run the main function
