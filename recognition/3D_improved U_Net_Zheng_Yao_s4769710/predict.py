# predict.py

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import unet_3d, dice_coefficient, combined_loss, dice_loss
from dataset import DataGenerator


# Load the trained model
model = tf.keras.models.load_model('best_model.h5', custom_objects={'combined_loss': combined_loss, 'dice_loss': dice_loss})

# Define parameters
input_dim = (128, 128, 64)
num_classes = 6

# selected one data at random, 4 is my lucky number.
test_image_file = 'Case_004_Week1_LFOV.nii.gz'
test_label_file = 'Case_004_Week1_SEMANTIC_LFOV.nii.gz'

# load and preprocess the image and label
img = nib.load(test_image_file).get_fdata()
lbl = nib.load(test_label_file).get_fdata()

# Instantiate DataGenerator to use its preprocessing methods
data_gen = DataGenerator([], [], dim=input_dim, num_classes=num_classes)
img = data_gen.preprocess_image(img)
lbl = data_gen.preprocess_label(lbl)

# Expand dimensions to match model input
X_test = np.expand_dims(img, axis=(0, -1))  # Shape: (1, dim[0], dim[1], dim[2], 1)

# Make prediction
y_pred = model.predict(X_test)

# Get predicted labels
y_pred = np.argmax(y_pred, axis=-1)  # Shape: (1, dim[0], dim[1], dim[2])
y_pred = np.squeeze(y_pred, axis=0)  # Remove batch dimension

# Visualize a slice
slice_idx = input_dim[2] // 2  # Middle slice
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img[:, :, slice_idx], cmap='gray')
plt.title('Input Image')

plt.subplot(1, 3, 2)
plt.imshow(lbl[:, :, slice_idx])
plt.title('Ground Truth')

plt.subplot(1, 3, 3)
plt.imshow(y_pred[:, :, slice_idx])
plt.title('Predicted Segmentation')

plt.savefig('prediction_example.png')
plt.show()

# Calculate Dice coefficient for this sample
y_true = lbl.astype(np.int32)
y_pred = y_pred.astype(np.int32)

dice_scores = []
for c in range(1, num_classes):  # Exclude background class 0
    y_true_c = (y_true == c).astype(np.float32)
    y_pred_c = (y_pred == c).astype(np.float32)
    dice = dice_coefficient(y_true_c, y_pred_c)
    dice_scores.append(dice)
    print(f'Class {c} Dice Score: {dice.numpy()}')

mean_dice = np.mean([d.numpy() for d in dice_scores])
print(f'Mean Dice Similarity Coefficient: {mean_dice}')
