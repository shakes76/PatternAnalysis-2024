# Import necessary libraries
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# Define data directories (update these paths as needed)
image_dir = r'D:\final\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon'
label_dir = r'D:\final\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon'

# Get sorted lists of image and label files
image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
label_files = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))

# Ensure that the number of images and labels is the same
assert len(image_files) == len(label_files), "Number of images and labels do not match."

# Define parameters
batch_size = 2  # Adjust based on your hardware capabilities
input_dim = (128, 128, 64)  # Adjusted input dimensions
num_classes = 6  # Number of classes (adjust based on your dataset)
epochs = 100  # Increased number of training epochs

# Data generator class with data augmentation and visualization
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
        X = np.empty((len(indexes), *self.dim, 1), dtype=np.float32)
        y = np.empty((len(indexes), *self.dim), dtype=np.uint8)

        # Generate data
        for i, idx in enumerate(indexes):
            # Load and preprocess image and label
            img = nib.load(self.image_files[idx]).get_fdata()
            lbl = nib.load(self.label_files[idx]).get_fdata()
            img = self.preprocess_image(img)
            lbl = self.preprocess_label(lbl)

            # Data augmentation
            img, lbl = self.augment(img, lbl)

            X[i, ..., 0] = img  # Add channel dimension
            y[i, ...] = lbl

            # Optional: Visualize samples (uncomment to use)
            # if index == 0 and i == 0:
            #     self.visualize_sample(img, lbl)

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
            angle = random.uniform(-5, 5)  # Reduced rotation angle
            image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
        return image, label

    def visualize_sample(self, img, lbl):
        import matplotlib.pyplot as plt
        slice_idx = self.dim[2] // 2
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, slice_idx], cmap='gray')
        plt.title('Preprocessed Image Slice')
        plt.subplot(1, 2, 2)
        plt.imshow(lbl[:, :, slice_idx])
        plt.title('Preprocessed Label Slice')
        plt.show()

# Build the 3D U-Net model with enhancements
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU

def unet_3d(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv3D(32, 3, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU()(c1)
    c1 = layers.Conv3D(32, 3, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU()(c1)
    p1 = layers.MaxPooling3D(2)(c1)
    p1 = Dropout(0.1)(p1)

    c2 = layers.Conv3D(64, 3, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU()(c2)
    c2 = layers.Conv3D(64, 3, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU()(c2)
    p2 = layers.MaxPooling3D(2)(c2)
    p2 = Dropout(0.1)(p2)

    c3 = layers.Conv3D(128, 3, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU()(c3)
    c3 = layers.Conv3D(128, 3, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU()(c3)
    p3 = layers.MaxPooling3D(2)(c3)
    p3 = Dropout(0.2)(p3)

    c4 = layers.Conv3D(256, 3, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU()(c4)
    c4 = layers.Conv3D(256, 3, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU()(c4)
    p4 = layers.MaxPooling3D(2)(c4)
    p4 = Dropout(0.2)(p4)

    # Bottleneck
    bn = layers.Conv3D(512, 3, padding='same')(p4)
    bn = BatchNormalization()(bn)
    bn = LeakyReLU()(bn)
    bn = layers.Conv3D(512, 3, padding='same')(bn)
    bn = BatchNormalization()(bn)
    bn = LeakyReLU()(bn)

    # Decoder
    u1 = layers.Conv3DTranspose(256, 2, strides=2, padding='same')(bn)
    u1 = layers.concatenate([u1, c4])
    u1 = Dropout(0.2)(u1)
    c5 = layers.Conv3D(256, 3, padding='same')(u1)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU()(c5)
    c5 = layers.Conv3D(256, 3, padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU()(c5)

    u2 = layers.Conv3DTranspose(128, 2, strides=2, padding='same')(c5)
    u2 = layers.concatenate([u2, c3])
    u2 = Dropout(0.2)(u2)
    c6 = layers.Conv3D(128, 3, padding='same')(u2)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU()(c6)
    c6 = layers.Conv3D(128, 3, padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU()(c6)

    u3 = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(c6)
    u3 = layers.concatenate([u3, c2])
    u3 = Dropout(0.1)(u3)
    c7 = layers.Conv3D(64, 3, padding='same')(u3)
    c7 = BatchNormalization()(c7)
    c7 = LeakyReLU()(c7)
    c7 = layers.Conv3D(64, 3, padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = LeakyReLU()(c7)

    u4 = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(c7)
    u4 = layers.concatenate([u4, c1])
    u4 = Dropout(0.1)(u4)
    c8 = layers.Conv3D(32, 3, padding='same')(u4)
    c8 = BatchNormalization()(c8)
    c8 = LeakyReLU()(c8)
    c8 = layers.Conv3D(32, 3, padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = LeakyReLU()(c8)

    outputs = layers.Conv3D(num_classes, 1, activation='softmax')(c8)

    model = keras.Model(inputs, outputs)
    return model

# Dice Loss function
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3, 4))
    return 1 - numerator / (denominator + 1e-6)

# Combined Loss Function
def combined_loss(y_true, y_pred):
    dl = dice_loss(y_true, y_pred)
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dl + cce

# Mean Dice Coefficient Metric
def mean_dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.cast(y_true, dtype=tf.float32)
    y_pred_f = tf.cast(tf.one_hot(tf.argmax(y_pred, axis=-1), depth=num_classes), dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(1,2,3,4))
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=(1,2,3,4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

# Compute class weights
from sklearn.utils import class_weight

# Note: Computing class weights over all data may be memory-intensive
# You might need to sample or compute them differently based on your dataset size
all_labels = []
for lbl_file in label_files:
    lbl = nib.load(lbl_file).get_fdata().flatten()
    all_labels.extend(lbl)
all_labels = np.array(all_labels)
classes = np.unique(all_labels)
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=all_labels)
class_weights_dict = dict(zip(classes, class_weights))

# Convert class weights to a list in order of class indices
class_weights_list = [class_weights_dict[i] if i in class_weights_dict else 1.0 for i in range(num_classes)]

# Weighted Categorical Crossentropy Loss
def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Clip prediction values to prevent division by zero
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        return tf.reduce_mean(loss)
    return loss

# Update the combined loss to include weighted categorical crossentropy
def combined_loss(y_true, y_pred):
    dl = dice_loss(y_true, y_pred)
    wce = weighted_categorical_crossentropy(class_weights_list)(y_true, y_pred)
    return dl + wce

# Parameters for K-Fold Cross-Validation
num_folds = 5  # Adjust based on your dataset size
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Arrays to store performance metrics
fold_dice_scores = []
fold_histories = []

# Combine images and labels into single lists for K-Fold
data = list(zip(image_files, label_files))
data = np.array(data)

# Start K-Fold Cross-Validation
for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
    print(f"\n\nProcessing Fold {fold + 1}/{num_folds}")

    # Split data for the current fold
    train_data = data[train_indices]
    val_data = data[val_indices]

    train_imgs = train_data[:, 0]
    train_labels = train_data[:, 1]
    val_imgs = val_data[:, 0]
    val_labels = val_data[:, 1]

    # Initialize data generators
    train_gen = DataGenerator(train_imgs, train_labels, batch_size=batch_size, dim=input_dim, num_classes=num_classes)
    val_gen = DataGenerator(val_imgs, val_labels, batch_size=batch_size, dim=input_dim, num_classes=num_classes)

    # Build and compile the model for each fold
    input_shape = (*input_dim, 1)
    model = unet_3d(input_shape, num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss=combined_loss,
                  metrics=[mean_dice_coefficient])

    # Define callbacks with the new monitoring metric
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_mean_dice_coefficient', patience=15, verbose=1, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_mean_dice_coefficient', factor=0.5, patience=7, verbose=1, mode='max'),
        # Optionally, use a learning rate scheduler
        # keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 0.9 ** epoch),
    ]

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    fold_histories.append(history)

    # Evaluate the model on the validation set
    val_loss, val_dice = model.evaluate(val_gen)
    print(f'Fold {fold + 1} Validation Loss: {val_loss}')
    print(f'Fold {fold + 1} Validation Mean Dice Coefficient: {val_dice}')

    # Calculate Dice scores on the validation set per class
    dice_scores = []
    for i in range(len(val_gen)):
        X_val, y_true = val_gen[i]
        y_pred = model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        for c in range(1, num_classes):  # Exclude background
            y_true_c = (y_true == c).astype(np.float32)
            y_pred_c = (y_pred == c).astype(np.float32)
            intersection = np.sum(y_true_c * y_pred_c)
            union = np.sum(y_true_c) + np.sum(y_pred_c)
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice)
            print(f'Fold {fold + 1}, Class {c} Dice Score: {dice}')

    mean_dice = np.mean(dice_scores)
    fold_dice_scores.append(mean_dice)
    print(f'Fold {fold + 1} Mean Dice Similarity Coefficient: {mean_dice}')

    # Optionally, save the model for this fold
    model.save(f'model_fold_{fold + 1}.keras')

    # Plot training and validation loss and mean dice coefficient for each fold
    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'fold_{fold + 1}_loss_plot.png')
    plt.close()

    # Plot Mean Dice Coefficient
    plt.figure()
    plt.plot(history.history['mean_dice_coefficient'], label='Training Mean Dice Coefficient')
    plt.plot(history.history['val_mean_dice_coefficient'], label='Validation Mean Dice Coefficient')
    plt.title(f'Fold {fold + 1} Mean Dice Coefficient During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Dice Coefficient')
    plt.legend()
    plt.savefig(f'fold_{fold + 1}_dice_plot.png')
    plt.close()

# Plot Mean Dice Similarity Coefficient across folds
plt.figure()
plt.plot(range(1, num_folds + 1), fold_dice_scores, marker='o')
plt.title('Mean Dice Similarity Coefficient Across Folds')
plt.xlabel('Fold')
plt.ylabel('Mean Dice Coefficient')
plt.savefig('mean_dice_coefficient_across_folds.png')
plt.close()

# Print overall mean Dice coefficient
overall_mean_dice = np.mean(fold_dice_scores)
print(f'Overall Mean Dice Similarity Coefficient: {overall_mean_dice}')
