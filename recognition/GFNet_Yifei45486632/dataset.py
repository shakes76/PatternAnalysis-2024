import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc

# Define the image size and batch size
IMAGE_SIZE = (224, 224)  # The normal size of GFNet is 224x224
BATCH_SIZE = 2

# Load and preprocess images using tf.data.Dataset
def load_images(directory):
    images = []
    labels = []
    print(f"\nLoading images from {directory}")
    
    # Iterate over each category folder in the directory
    for label in os.listdir(directory):
        label_folder = os.path.join(directory, label)
        if os.path.isdir(label_folder):
            # Get all the image files in that category
            image_files = [f for f in os.listdir(label_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} images in {label} class")
            
            # Add a path and label to each image
            for image_file in image_files:
                file_path = os.path.join(label_folder, image_file)
                images.append(file_path)
                labels.append(label)
    
    print(f"Total images loaded: {len(images)}")
    # Print the actual label distribution
    label_distribution = {}
    for label in labels:
        label_distribution[label] = label_distribution.get(label, 0) + 1
    print("Label distribution:")
    for label, count in label_distribution.items():
        print(f"{label}: {count} images")
        
    return images, labels

# TensorFlow Dataloader - to pre-process images
@tf.function
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
def build_dataset(image_paths, labels, is_training=True):
    print(f"Building dataset with {len(image_paths)} images and {len(labels)} labels")
    
    # Convert labels to numbers
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in labels]
    
    print(f"Label mapping: {label_to_index}")
    print(f"Unique labels: {unique_labels}")
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, numerical_labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Obtaining the test set
def get_test_dataset():
    print("\nLoading test dataset...")
    test_images, test_labels = load_images('test')
    print(f"Loaded {len(test_images)} test images")
    test_dataset = build_dataset(test_images, test_labels, is_training=False)
    return test_dataset

# Split and obtain training and validation sets - each class should have a uniform and random distribution
def get_train_validation_dataset():
    print("\nLoading training and validation datasets...")
    train_images, train_labels = load_images('train')
    
    print("\nPerforming train-validation split...")
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, 
        train_labels,
        test_size=0.15, 
        random_state=42, 
        stratify=train_labels
    )
    
    print(f"After split:")
    print(f"Training set size: {len(train_images)} images")
    print(f"Validation set size: {len(val_images)} images")
    
    train_distribution = {}
    for label in train_labels:
        train_distribution[label] = train_distribution.get(label, 0) + 1
    print("\nTraining set distribution:")
    for label, count in train_distribution.items():
        print(f"{label}: {count} images")
    
    print("\nCreating TensorFlow datasets...")
    train_dataset = build_dataset(train_images, train_labels, is_training=True)
    val_dataset = build_dataset(val_images, val_labels, is_training=False)
    
    return train_dataset, val_dataset

# Get the images and labels in dataset after preprocessing
def extract_from_dataset(dataset):
    images_list = []
    labels_list = []
    total_batches = 0
    batch_size = 32
    
    try:
        print("Starting data extraction...")
        for batch_images, batch_labels in dataset:
            # Transform the current batch and add it to the list
            images_list.append(batch_images.numpy())
            labels_list.append(batch_labels.numpy())
            total_batches += 1
            
            # The data is merged and the memory is cleaned every 32 batches
            if len(images_list) >= batch_size:
                print(f"Processed {total_batches} batches...")
                # Merging the current batch
                interim_images = np.concatenate(images_list, axis=0)
                interim_labels = np.concatenate(labels_list, axis=0)
                
                # Clear the list and keep only the merged array
                images_list = [interim_images]
                labels_list = [interim_labels]
                
                # Clear up memory
                tf.keras.backend.clear_session()
                gc.collect()
    
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise
    
    print("Final concatenation...")
    all_images = np.concatenate(images_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    print(f"Extraction complete. Final shapes - Images: {all_images.shape}, Labels: {all_labels.shape}")
    return all_images, all_labels

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
