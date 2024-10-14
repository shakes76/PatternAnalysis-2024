import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the images and basic data preprocessing
def load_images(directory):
    images = []
    labels = []
    # Read the image data and convert it into an array to store the labels associated with the image
    for label in os.listdir(directory):
        label_folder =os.path.join(directory, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                file_path =os.path.join(label_folder,filename)
                try:
                    img = load_img(file_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return images,labels

# Get images and labels for test
def get_test_dataset():
    test_images,test_labels = load_images('test')
    return test_images, test_labels

# Split and get images and labels for train and validation
def get_train_validation_dataset():
    train_images, train_labels = load_images('train')
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    return train_images, val_images, train_labels, val_labels

# Test the result
if __name__ == "__main__":
    train_images, val_images, train_labels, val_labels = get_train_validation_dataset()
    test_images,test_labels = get_test_dataset()

    print(f"Loaded {len(train_images)} training images.")
    print(f"Shape of the image:{train_images[0].shape}")
    print(f"Total number of training labels: {len(train_labels)}")

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i +1)
        plt.imshow(np.uint8(train_images[i]))
        plt.title(train_labels[i])
        plt.axis('off')
    plt.show()