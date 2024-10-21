# Import necessary libraries
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# Define data directories (update these paths as needed)
image_dir = r'D:\path\to\image\directory'
label_dir = r'D:\path\to\label\directory'

# Get sorted lists of image and label files
image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
label_files = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))

# Ensure that the number of images and labels is the same
assert len(image_files) == len(label_files), "Number of images and labels do not match."

# Split data into training, validation, and test sets
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42)

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    train_imgs, train_labels, test_size=0.1, random_state=42)  # 10% of training data for validation

# Define parameters
batch_size = 1
input_dim = (128, 128, 64)  # Adjusted input dimensions
num_classes = 6  # Number of classes (adjust based on your dataset)
epochs = 50  # Increased number of training epochs


# Data generator class with data augmentation
class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_files, label_files, batch_size=1, dim=(128, 128, 64), num_classes=6, shuffle=True):
        self.image_files = image_files
        self.label_files = label_files
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize
        X = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        # Generate data
        for i, idx in enumerate(indexes):
            # Load and preprocess image
            img = nib.load(self.image_files[idx]).get_fdata()
            lbl = nib.load(self.label_files[idx]).get_fdata()
            img = self.preprocess_image(img)
            lbl = self.preprocess_label(lbl)

            # Data augmentation
            img, lbl = self.augment(img, lbl)

            X[i, ..., 0] = img  # Add channel dimension
            y[i, ...] = lbl

        # One-hot encode labels
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_image(self, img):
        # Normalize the image
        img = (img - np.mean(img)) / np.std(img)
        # Resize to target dimensions
        img = self.resize_volume(img)
        return img

    def preprocess_label(self, lbl):
        lbl = lbl.astype(np.uint8)
        lbl = self.resize_volume(lbl, interpolation='nearest')
        return lbl

    def resize_volume(self, volume, interpolation='linear'):
        # Resize across z-axis (depth)
        from scipy.ndimage import zoom
        depth_factor = self.dim[2] / volume.shape[2]
        width_factor = self.dim[0] / volume.shape[0]
        height_factor = self.dim[1] / volume.shape[1]
        if interpolation == 'linear':
            volume = zoom(volume, (width_factor, height_factor, depth_factor), order=1)
        elif interpolation == 'nearest':
            volume = zoom(volume, (width_factor, height_factor, depth_factor), order=0)
        return volume

    def augment(self, image, label):
        # Random flip
        if random.random() > 0.5:
            axis = random.choice([0, 1])
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
        return image, label


# Build the 3D U-Net model
def unet_3d(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv3D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D(2)(c1)

    c2 = layers.Conv3D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D(2)(c2)

    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling3D(2)(c3)

    c4 = layers.Conv3D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling3D(2)(c4)

    # Bottleneck
    bn = layers.Conv3D(256, 3, activation='relu', padding='same')(p4)
    bn = layers.Conv3D(256, 3, activation='relu', padding='same')(bn)

    # Decoder
    u1 = layers.Conv3DTranspose(128, 2, strides=2, padding='same')(bn)
    u1 = layers.concatenate([u1, c4])
    c5 = layers.Conv3D(128, 3, activation='relu', padding='same')(u1)
    c5 = layers.Conv3D(128, 3, activation='relu', padding='same')(c5)

    u2 = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(c5)
    u2 = layers.concatenate([u2, c3])
    c6 = layers.Conv3D(64, 3, activation='relu', padding='same')(u2)
    c6 = layers.Conv3D(64, 3, activation='relu', padding='same')(c6)

    u3 = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(c6)
    u3 = layers.concatenate([u3, c2])
    c7 = layers.Conv3D(32, 3, activation='relu', padding='same')(u3)
    c7 = layers.Conv3D(32, 3, activation='relu', padding='same')(c7)

    u4 = layers.Conv3DTranspose(16, 2, strides=2, padding='same')(c7)
    u4 = layers.concatenate([u4, c1])
    c8 = layers.Conv3D(16, 3, activation='relu', padding='same')(u4)
    c8 = layers.Conv3D(16, 3, activation='relu', padding='same')(c8)

    outputs = layers.Conv3D(num_classes, 1, activation='softmax')(c8)

    model = keras.Model(inputs, outputs)
    return model


# Custom loss functions
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3, 4))
    return 1 - tf.reduce_mean(numerator / (denominator + 1e-6))


def combined_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return ce_loss + dl


# Calculate class weights to handle class imbalance
def calculate_class_weights(labels):
    # Flatten labels to compute class frequencies
    all_labels = np.concatenate([np.ravel(nib.load(f).get_fdata()) for f in labels])
    class_weights = {}
    classes = np.unique(all_labels)
    total = len(all_labels)
    for c in classes:
        count = np.sum(all_labels == c)
        class_weights[c] = total / (len(classes) * count)
    return class_weights


# Get class weights
class_weights = calculate_class_weights(train_labels)

# Convert class weights to a list
weights_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
weights_tensor = tf.constant(weights_list, dtype=tf.float32)


def weighted_categorical_crossentropy(y_true, y_pred):
    # Compute the categorical cross-entropy loss
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # Multiply by class weights
    weights = tf.reduce_sum(y_true * weights_tensor, axis=-1)
    cce = cce * weights
    return tf.reduce_mean(cce)


def weighted_combined_loss(y_true, y_pred):
    return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# Build and compile the model
input_shape = (*input_dim, 1)  # Add channel dimension
model = unet_3d(input_shape, num_classes)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=weighted_combined_loss,
              metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',  # Updated file extension
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
test_gen = DataGenerator(test_imgs, test_labels, batch_size=batch_size, dim=input_dim, num_classes=num_classes,
                         shuffle=False)
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')


# Calculate Dice Similarity Coefficient on test set
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)



dice_scores = []

for i in range(len(test_gen)):
    X_test, y_true = test_gen[i]
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    for c in range(1, num_classes):  # Exclude background
        y_true_c = (y_true == c).astype(np.float32)
        y_pred_c = (y_pred == c).astype(np.float32)
        dice = dice_coefficient(y_true_c, y_pred_c)
        dice_scores.append(dice)
        print(f'Class {c} Dice Score: {dice.numpy()}')

# Calculate mean Dice score across all classes
mean_dice = np.mean([d.numpy() for d in dice_scores])
print(f'Mean Dice Similarity Coefficient: {mean_dice}')
