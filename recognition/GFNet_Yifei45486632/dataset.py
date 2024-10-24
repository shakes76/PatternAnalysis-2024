import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the image size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32  # You can adjust this according to your GPU memory

# Load and preprocess images using tf.data.Dataset
def load_images_with_labels(directory):
    images = []
    labels = []
    # Read the image data and convert it into an array to store the labels associated with the image
    for label in os.listdir(directory):
        label_folder = os.path.join(directory, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                file_path = os.path.join(label_folder, filename)
                try:
                    images.append(file_path)  # Store file paths instead of image data
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return images, labels

# TensorFlow data loader
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)  # Resize the image
    img = img / 255.0  # Normalize pixel values
    return img, label

def build_dataset(image_paths, labels, shuffle=True):
    # Convert string labels to numerical indices
    unique_labels = list(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in labels]

    # Create a TensorFlow Dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, numerical_labels))
    
    # Map the dataset to apply the preprocess_image function
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Batch and prefetch for better performance
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Get the test dataset
def get_test_dataset():
    test_images, test_labels = load_images_with_labels('test')
    test_dataset = build_dataset(test_images, test_labels, shuffle=False)
    return test_dataset

# Split and get the train and validation datasets
def get_train_validation_dataset():
    train_images, train_labels = load_images_with_labels('train')
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    train_dataset = build_dataset(train_images, train_labels)
    val_dataset = build_dataset(val_images, val_labels, shuffle=False)
    return train_dataset, val_dataset

# Test the data loading and visualization
if __name__ == "__main__":
    train_dataset, val_dataset = get_train_validation_dataset()
    test_dataset = get_test_dataset()

    # Display a batch of images
    for images, labels in train_dataset.take(1):  # Just take one batch of images
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis('off')
        plt.show()
