import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

#Function to calculate the Dice Similarity Coefficient (DSC)
#Source: https://github.com/mohammadbehzadpour/Brain-MRI-Segmentation-Using-Unet-2D-Keras/blob/main/Unet-Brain-MRI-Segmentation-Tensorflow-Keras.ipynb
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = np.ravel(y_true)  # Flatten the arrays
    y_pred_f = np.ravel(y_pred)  # Flatten the arrays
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

#Load the saved model
model = load_model('unet_final_model.keras')

#Load the test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

#Reshape test data to add the channel dimension (if needed)
X_test = np.expand_dims(X_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

#Normalise the test data
X_test = X_test / 255.0
y_test = y_test / 3
y_test = np.where(y_test > 0.5, 1, 0)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Binarise predictions to match ground truth format (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(np.float32)

#Calculate Dice Similarity Coefficient for each image and the overall score
dice_scores = [dice_coefficient(y_test[i], y_pred_binary[i]) for i in range(len(y_test))]
overall_dice = np.mean(dice_scores)

#Print the overall Dice Similarity Coefficient
print(f'Overall Dice Similarity Coefficient on Test Set: {overall_dice:.4f}')

#Now, let's generate a mask with our model and compare it to a true mask
sample_index = 1
input_image = X_test[sample_index]
true_mask = y_test[sample_index]
predicted_mask = y_pred_binary[sample_index]

#Plot the input image, true mask, and predicted mask
plt.figure(figsize=(12, 4))

#Plot input image
plt.subplot(1, 3, 1)
plt.imshow(input_image[:, :, 0], cmap='gray')
plt.title('Input Image')
plt.axis('off')

#Plot true mask 
plt.subplot(1, 3, 2)
plt.imshow(true_mask[:, :, 0], cmap='gray')
plt.title('True Mask')
plt.axis('off')

#Plot predicted mask
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask[:, :, 0], cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

#Save the plot to a file
output_file = 'test_image_prediction_comparison_100.png'
plt.savefig(output_file)

plt.show()
