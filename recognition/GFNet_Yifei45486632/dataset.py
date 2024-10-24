import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the image size and batch size
IMAGE_SIZE = (224, 224)  # The normal size of GFNet is 224x224
BATCH_SIZE = 8

# Load and preprocess images using tf.data.Dataset
def load_images(directory):
    images = []
    labels = []
    # Traversing directories
    for label in os.listdir(directory):
        label_folder = os.path.join(directory, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                file_path = os.path.join(label_folder, filename)
                images.append(file_path)  # Save file path
                labels.append(label)      # Save labels
    return images, labels

# TensorFlow Dataloader - to pre-process images
def preprocess_image(file_path, label):
    # Definition of image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)  # Resize the size of image to fit the input of GFNet model
    # Definition of label
    label = tf.cast(label, tf.int32)
    img = img / 255.0  # Normalized pixel values
    return img, label

# Build the dataset used in GFNet
def build_dataset(image_paths, labels, shuffle=True):
    # Converts string labels to numeric indices
    unique_labels = list(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in labels]

    # Create TensorFlow datasets from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, numerical_labels))
    
    # Map the dataset to apply a preprocessing function
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # Batching and prefetching to improve performance
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Obtaining the test set
def get_test_dataset():
    test_images, test_labels = load_images('test')
    # test_dataset = build_dataset(test_images, test_labels, shuffle=False)
    test_dataset = build_dataset(test_images, test_labels)
    return test_dataset

# Split and obtain training and validation sets - each class should have a uniform and random distribution
def get_train_validation_dataset():
    train_images, train_labels = load_images('train')

    # Convert he string label into a numerical index, and the random division makes it non-biased
    unique_labels = list(set(train_labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in train_labels]

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, numerical_labels, test_size=0.15, random_state=42, stratify=numerical_labels 
    )
    train_dataset = build_dataset(train_images, train_labels)
    val_dataset = build_dataset(val_images, val_labels, shuffle=False)
    return train_dataset, val_dataset

# Get the labels in dataset after preprocessing
def extract_labels_from_dataset(dataset):
    all_labels = []
    for _, labels in dataset.unbatch():
        all_labels .append(labels.numpy())
    return np.array(all_labels)

# Test the functionality of the data loader
if __name__ == "__main__":
    train_dataset, val_dataset = get_train_validation_dataset()
    test_dataset = get_test_dataset()

    # Display a batch of images
    for images, labels in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.show()
