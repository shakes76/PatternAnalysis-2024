import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from dataset import load_data
from train import dice_coefficient, train_unet_model, n_classes
from modules import unet_model, unet_model1


images_train, images_test, images_validate, images_seg_test, images_seg_train, images_seg_validate = load_data()


#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def calculate_dice_per_class(y_true, y_pred, num_classes):
    """
    Calculate Dice coefficient for each class.

    Args:
        y_true: Ground truth segmentation masks (one-hot encoded).
        y_pred: Predicted segmentation masks (one-hot encoded).
        num_classes: Number of classes.

    Returns:
        dice_scores: List of Dice coefficients for each class.
    """
    dice_scores = []

    for class_id in range(num_classes):
        true_class = (y_true == class_id).astype(np.float32)
        pred_class = (y_pred == class_id).astype(np.float32)

        dice = dice_coefficient(true_class, pred_class)
        dice_scores.append(dice.numpy())  # Convert to NumPy value for easy handling

    return dice_scores

# Function to save a validation image
def save_validation_image(image, mask, prediction, index):
    """Saves the original image, mask, and prediction."""
    
    # If the mask has more than one channel, convert it back to a single channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Similarly, for predictions
    if len(prediction.shape) == 4:
        prediction = prediction[:, :, :, 0]

    # Create a figure
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze())  
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask)  
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(prediction)  
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # Save the figure
    plt.savefig(f'validation_image_{index}.png')
    plt.close()





# Initialize the U-Net model
model = unet_model1(n_classes, input_size=(256, 128, 1))

# Train the U-Net model
history = train_unet_model(model, images_train, images_seg_train, 
                           images_validate, images_seg_validate, 
                           model_save_path="best_unet_model.h5")
                           #class_weights=class_weights)



index = 0
image = images_test[index]  # Shape (256, 128, 1)
mask = images_seg_test[index]   # Shape (256, 128, 5)



# Get prediction from the model
prediction = model.predict(image[np.newaxis, ..., np.newaxis])  # shape of input (1, 256, 128, 1)

# Convert prediction to class labels (argmax along the last axis)
predicted_labels = np.argmax(prediction[0], axis=-1)  # Shape (256, 128), per-pixel class

# Convert one-hot encoded mask to class labels for comparison/visualization
true_labels = np.argmax(mask, axis=-1) 

# Save the images (modify this function to handle the labels as needed)
save_validation_image(image, true_labels, predicted_labels, index)




images_test_predict = np.expand_dims(images_test, axis=-1)  # Adds the channel dimension if missing
# Make predictions on the test set
predictions = model.predict(images_test_predict)

# Convert predictions to class labels (argmax along the last axis)
predicted_labels = np.argmax(predictions, axis=-1)  # Shape (num_samples, height, width)

# Convert one-hot encoded masks to class labels for comparison
true_labels = np.argmax(images_seg_test, axis=-1)  # Shape (num_samples, height, width)

# Calculate Dice coefficients for each class
dice_scores = calculate_dice_per_class(true_labels, predicted_labels, n_classes)

# Print the Dice coefficients for each class
for class_id, score in enumerate(dice_scores):
    print(f"Dice Coefficient for Class {class_id}: {score:.4f}")