import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix



from util import calculate_dice_per_class, n_classes, run, save_validation_image
from dataset import load_data


images_train, images_test, images_validate, images_seg_test, images_seg_train, images_seg_validate = load_data()

#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def print_image(index):
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


#Load the model
model = tf.keras.models.load_model('best_unet_model_drop0.2.h5', compile=False)

for i in range(5):
    print_image(i*10)

images_test_predict = np.expand_dims(images_test, axis=-1)  # Adds the channel dimension

# Make predictions on the test set
predictions = model.predict(images_test_predict)

# Convert predictions to class labels (argmax along the last axis)
predicted_labels = np.argmax(predictions, axis=-1) 

# Convert one-hot encoded masks to class labels for comparison
true_labels = np.argmax(images_seg_test, axis=-1) 

# Calculate Dice coefficients for each class
dice_scores = calculate_dice_per_class(true_labels, predicted_labels, n_classes)

# Print the Dice coefficients for each class
for class_id, score in enumerate(dice_scores):
    print(f"Dice Coefficient for Class {class_id}: {score:.4f}")

