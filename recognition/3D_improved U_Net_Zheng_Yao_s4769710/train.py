import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from modules import unet_3d, combined_loss
from dataset import DataGenerator

image_dir = r'D:\final\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon'
label_dir = r'D:\final\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon'

image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
label_files = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))

# Split data into training, validation, and test sets
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42)

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    train_imgs, train_labels, test_size=0.1, random_state=42)  # 10% of training data for validation

# Define parameters
batch_size = 2
input_dim = (128, 128, 64)
num_classes = 6  # Run a code trough all the data and found out the value 6.
epochs = 100

# Build and compile the model
input_shape = (*input_dim, 1)
model = unet_3d(input_shape, num_classes)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_loss,
              metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    ),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
]

# Initialize data generators
train_gen = DataGenerator(train_imgs, train_labels, batch_size=batch_size, dim=input_dim, num_classes=num_classes)
val_gen = DataGenerator(val_imgs, val_labels, batch_size=batch_size, dim=input_dim, num_classes=num_classes)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks
)

# Plot losses and metrics
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

# Evaluate the model on the test set
test_gen = DataGenerator(test_imgs, test_labels, batch_size=batch_size, dim=input_dim, num_classes=num_classes, shuffle=False)
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Calculate Dice Similarity Coefficient on test set
from modules import dice_coefficient

dice_scores = []

for i in range(len(test_gen)):
    X_test, y_true = test_gen[i]
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    for c in range(1, num_classes):  # Exclude background class 0
        y_true_c = (y_true == c).astype(np.float32)
        y_pred_c = (y_pred == c).astype(np.float32)
        dice = dice_coefficient(y_true_c, y_pred_c)
        dice_scores.append(dice)
        print(f'Class {c} Dice Score: {dice.numpy()}')

# Calculate mean Dice score across all classes
mean_dice = np.mean([d.numpy() for d in dice_scores])
print(f'Mean Dice Similarity Coefficient: {mean_dice}')
